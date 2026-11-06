use crate::OutputFileData;
use crate::alignment;
use crate::alignment::Alignment;
use crate::bail;
use crate::coff;
use crate::coff::Coff;
use crate::coff::CoffRelocation;
use crate::coff::DosHeader;
use crate::coff::output_section_id;
use crate::coff::part_id;
use crate::debug_assert_bail;
use crate::ensure;
use crate::error::Context;
use crate::error::Result;
use crate::file_writer::SizedOutput;
use crate::file_writer::excessive_allocation;
use crate::file_writer::insufficient_allocation;
use crate::file_writer::split_buffers_by_alignment;
use crate::file_writer::split_output_by_group;
use crate::file_writer::split_output_into_sections;
use crate::layout::EpilogueLayout;
use crate::layout::FileLayout;
use crate::layout::HeaderInfo;
use crate::layout::InternalSymbols;
use crate::layout::Layout;
use crate::layout::ObjectLayout;
use crate::layout::OutputRecordLayout;
use crate::layout::PreludeLayout;
use crate::layout::Resolution;
use crate::layout::Section;
use crate::layout::SymbolCopyInfo;
use crate::layout::SyntheticSymbolsLayout;
use crate::output_kind::OutputKind;
use crate::output_section_id::OrderEvent;
use crate::output_section_id::OutputSectionId;
use crate::output_section_id::OutputSections;
use crate::output_section_id::SectionOutputInfo;
use crate::output_section_map::OutputSectionMap;
use crate::output_section_part_map::OutputSectionPartMap;
use crate::output_trace::HexU64;
use crate::output_trace::TraceOutput;
use crate::platform;
use crate::platform::Arch;
use crate::platform::Args;
use crate::platform::ObjectFile;
use crate::platform::Relocation;
use crate::platform::RelocationSequence;
use crate::platform::SectionHeader as _;
use crate::platform::Symbol as _;
use crate::resolution::SectionSlot;
use crate::sharding::ShardKey;
use crate::symbol_db::SymbolDb;
use crate::symbol_db::SymbolId;
use crate::timing::timing_phase;
use crate::timing::verbose_timing_phase;
use crate::value_flags::PerSymbolFlags;
use crate::value_flags::ValueFlags;
use linker_utils::elf::RelocationKind;
use linker_utils::elf::RelocationSize;
use linker_utils::utils::slice_from_all_bytes_mut;
use object::LittleEndian;
use object::SymbolIndex;
use object::from_bytes_mut;
use object::pe::ImageFileHeader;
use object::pe::ImageOptionalHeader64;
use object::pe::SymbolSection;
use object::write::WritableBuffer;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator as _;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::atomic::Ordering::Relaxed;
use tracing::debug_span;

type CoffLayout<'data> = Layout<'data, Coff>;

pub(crate) fn write<'data, A: Arch<Platform = Coff>>(
    sized_output: &mut SizedOutput<impl OutputFileData>,
    layout: &CoffLayout<'data>,
) -> Result {
    write_file_contents::<A>(sized_output, layout)?;
    // TODO: elf specific right now
    // if layout.args().common().validate_output {
    //     crate::validation::validate_bytes(layout, &sized_output.out)?;
    // }

    Ok(())
}

fn write_file_contents<'data, A: Arch<Platform = Coff>>(
    sized_output: &mut SizedOutput<impl OutputFileData>,
    layout: &CoffLayout<'data>,
) -> Result {
    timing_phase!("Write data to file");
    let (mut section_buffers, padding) = split_output_into_sections(layout, &mut sized_output.out);

    fill_padding_for_sections::<A>(layout, padding);

    let mut writable_buckets = split_buffers_by_alignment(&mut section_buffers, layout);
    let groups_and_buffers = split_output_by_group(layout, &mut writable_buckets);
    groups_and_buffers
        .into_par_iter()
        .with_max_len(1)
        .try_for_each(|(group, mut buffers)| -> Result {
            verbose_timing_phase!("Write group");

            let mut table_writer =
                TableWriter::from_layout(layout, group.strtab_start_offset, &mut buffers);

            for file in &group.files {
                write_file::<A>(
                    file,
                    &mut buffers,
                    &mut table_writer,
                    layout,
                    &sized_output.trace,
                )
                .with_context(|| format!("Failed copying from {file} to output file"))?;
            }
            table_writer
                .validate_empty(&group.mem_sizes)
                .with_context(|| format!("validate_empty failed for {group}"))?;
            Ok(())
        })?;

    for (output_section_id, _) in layout.output_sections.ids_with_info() {
        let relocations = layout
            .relocation_statistics
            .get(output_section_id)
            .load(Relaxed);

        if relocations > 0 {
            tracing::debug!(
                target: "metrics",
                section = layout.output_sections.display_name(output_section_id),
                relocations, "resolved relocations");
        }
    }

    fill_padding(section_buffers);

    Ok(())
}

struct TableWriter<'layout, 'out> {
    _output_kind: OutputKind,
    symbol_table_writer: SymbolTableWriter<'layout, 'out>,
    section_headers: &'out mut [coff::SectionHeader],
}

impl<'layout, 'out> TableWriter<'layout, 'out> {
    fn from_layout(
        layout: &'layout CoffLayout,
        strtab_start_offset: u32,
        buffers: &mut OutputSectionPartMap<&'out mut [u8]>,
    ) -> TableWriter<'layout, 'out> {
        let symbol_table_writer = SymbolTableWriter::new(
            strtab_start_offset,
            buffers,
            &layout.output_sections,
            &layout.section_layouts,
        );

        Self::new(layout.symbol_db.output_kind, buffers, symbol_table_writer)
    }

    fn new(
        output_kind: OutputKind,
        buffers: &mut OutputSectionPartMap<&'out mut [u8]>,
        symbol_table_writer: SymbolTableWriter<'layout, 'out>,
    ) -> TableWriter<'layout, 'out> {
        TableWriter {
            _output_kind: output_kind,
            symbol_table_writer,
            section_headers: slice_from_all_bytes_mut(buffers.take(part_id::SECTION_HEADERS)),
        }
    }

    fn take_section_header(&mut self) -> Result<&'out mut coff::SectionHeader> {
        self.section_headers
            .split_off_first_mut()
            .ok_or_else(|| insufficient_allocation("section headers"))
    }

    /// Checks that we used all of the entries that we requested during layout.
    fn validate_empty(&self, mem_sizes: &OutputSectionPartMap<u64>) -> Result {
        if !self.section_headers.is_empty() {
            return Err(excessive_allocation(
                "section headers",
                size_of_val(self.section_headers) as u64,
                mem_sizes.get(part_id::SECTION_HEADERS),
            ));
        }

        self.symbol_table_writer.check_exhausted()?;

        Ok(())
    }
}

fn fill_padding_for_sections<A: Arch<Platform = Coff>>(
    layout: &Layout<'_, Coff>,
    padding: crate::file_writer::PaddingSlices<'_>,
) {
    timing_phase!("Fill padding for sections");

    for pslice in padding.slices {
        if let Some(section_id) = pslice.parent_section_id {
            let section_info = layout.output_sections.output_info(section_id);
            fill_section_padding::<A>(pslice.slice, section_info);
        } else {
            pslice.slice.fill(0);
        }
    }
}

fn fill_padding(mut section_buffers: OutputSectionMap<&mut [u8]>) {
    section_buffers.for_each_mut(|_, out| {
        out.fill(0);
    });
}

fn populate_dos_header<A: Arch<Platform = Coff>>(
    layout: &CoffLayout,
    header: &mut DosHeader,
) -> Result {
    *header = DosHeader {
        e_magic: object::pe::IMAGE_DOS_SIGNATURE.into(),
        e_cblp: Default::default(),
        e_cp: Default::default(),
        e_crlc: Default::default(),
        e_cparhdr: u16::try_from(size_of::<DosHeader>() / u16::BITS as usize)?.into(),
        e_minalloc: Default::default(),
        e_maxalloc: Default::default(),
        e_ss: Default::default(),
        e_sp: Default::default(),
        e_csum: Default::default(),
        e_ip: Default::default(),
        e_cs: Default::default(),
        e_lfarlc: Default::default(),
        e_ovno: Default::default(),
        e_res: [0.into(); 4],
        e_oemid: Default::default(),
        e_oeminfo: Default::default(),
        e_res2: [0.into(); 10],
        e_lfanew: u32::try_from(
            layout
                .section_layouts
                .get(output_section_id::NT_HEADER)
                .file_offset,
        )?
        .into(),
    };

    Ok(())
}

fn write_dos_stub<A: Arch<Platform = Coff>>(mut bytes: &mut [u8]) {
    bytes.write_bytes(coff::DOS_STUB);
}

fn populate_nt_header<A: Arch<Platform = Coff>>(
    layout: &CoffLayout,
    header_info: &HeaderInfo,
    header: &mut coff::NtHeader,
) -> Result {
    let output_kind = layout.symbol_db.output_kind;
    let mut characteristics =
        object::pe::IMAGE_FILE_EXECUTABLE_IMAGE | object::pe::IMAGE_FILE_LARGE_ADDRESS_AWARE;
    if output_kind.is_shared_object() {
        characteristics |= object::pe::IMAGE_FILE_DLL;
    }

    let first_regular_section = layout.section_layouts.get(output_section_id::TEXT);

    let file_header = ImageFileHeader {
        machine: object::pe::IMAGE_FILE_MACHINE_AMD64.into(),
        number_of_sections: u16::try_from(header_info.num_output_sections_with_content)?.into(),
        time_date_stamp: Default::default(),
        pointer_to_symbol_table: u32::try_from(
            layout
                .section_layouts
                .get(output_section_id::SYMTAB)
                .file_offset,
        )?
        .into(),
        number_of_symbols: u32::try_from(
            layout
                .section_layouts
                .get(output_section_id::SYMTAB)
                .file_size
                / object::pe::IMAGE_SIZEOF_SYMBOL,
        )?
        .into(),
        size_of_optional_header: coff::SIZE_OF_OPTIONAL_HEADER64.into(),
        characteristics: characteristics.into(),
    };
    // TODO: Skips a lot of logic compared to ELF one
    let entry_point_address = u32::try_from(rva(
        layout.resolved_entry_symbol_address()?.unwrap(),
        layout,
    ))?;
    let size_of_code = u32::try_from(
        layout
            .section_layouts
            .get(output_section_id::TEXT)
            .file_size,
    )?
    .into();
    let size_of_initialized_data = u32::try_from(
        layout
            .output_sections
            .ids_with_info()
            .map(|(id, sec)| {
                if sec
                    .section_attributes
                    .flags
                    .contains(object::pe::IMAGE_SCN_CNT_INITIALIZED_DATA)
                {
                    layout.section_layouts.get(id).file_size
                } else {
                    0
                }
            })
            .sum::<usize>(),
    )?
    .into();
    let size_of_uninitialized_data =
        u32::try_from(layout.section_layouts.get(output_section_id::BSS).file_size)?.into();
    let optional_header = ImageOptionalHeader64 {
        magic: object::pe::IMAGE_NT_OPTIONAL_HDR64_MAGIC.into(),
        major_linker_version: 14,
        minor_linker_version: 0,
        size_of_code,
        size_of_initialized_data,
        size_of_uninitialized_data,
        address_of_entry_point: entry_point_address.into(),
        base_of_code: u32::try_from(rva(first_regular_section.mem_offset, layout))?.into(),
        image_base: layout.args().image_base().unwrap().into(),
        // TODO: Create const
        section_alignment: u32::try_from(Alignment::from_exponent(12)?.value())?.into(),
        // Wild doesn't support separate memory and file alignment
        file_alignment: u32::try_from(first_regular_section.alignment.value())?.into(),
        major_operating_system_version: 6.into(),
        minor_operating_system_version: Default::default(),
        major_image_version: Default::default(),
        minor_image_version: Default::default(),
        major_subsystem_version: 6.into(),
        minor_subsystem_version: Default::default(),
        win32_version_value: Default::default(),
        size_of_image: u32::try_from(
            first_regular_section.alignment.align_up(rva(
                layout
                    .section_layouts
                    .iter()
                    .map(|(_, sec)| sec.mem_end())
                    .max()
                    .unwrap(),
                layout,
            )),
        )?
        .into(),
        size_of_headers: u32::try_from(
            first_regular_section.alignment.align_up(
                layout
                    .section_layouts
                    .get(output_section_id::DOS_STUB)
                    .mem_size
                    + layout
                        .section_layouts
                        .get(output_section_id::NT_HEADER)
                        .mem_size
                    + layout
                        .section_layouts
                        .get(output_section_id::SECTION_HEADERS)
                        .mem_size,
            ),
        )?
        .into(),
        check_sum: Default::default(),
        subsystem: object::pe::IMAGE_SUBSYSTEM_WINDOWS_CUI.into(),
        dll_characteristics: Default::default(),
        size_of_stack_reserve: Default::default(),
        size_of_stack_commit: Default::default(),
        size_of_heap_reserve: Default::default(),
        size_of_heap_commit: Default::default(),
        loader_flags: Default::default(),
        number_of_rva_and_sizes: u32::try_from(object::pe::IMAGE_NUMBEROF_DIRECTORY_ENTRIES)?
            .into(),
    };

    header.signature = object::pe::IMAGE_NT_SIGNATURE.into();
    header.file_header = file_header;
    header.optional_header = optional_header;

    Ok(())
}

fn write_file<'data, A: Arch<Platform = Coff>>(
    file: &FileLayout<'data, Coff>,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    table_writer: &mut TableWriter<'_, '_>,
    layout: &CoffLayout<'data>,
    trace: &TraceOutput,
) -> Result {
    match file {
        FileLayout::Object(s) => {
            write_object::<A>(s, buffers, table_writer, layout, trace)?;
        }
        FileLayout::Prelude(s) => {
            write_prelude::<A>(s, buffers, table_writer, layout)?;
        }
        FileLayout::Epilogue(s) => write_epilogue::<A>(s, buffers, table_writer, layout, trace),
        FileLayout::SyntheticSymbols(s) => {
            write_synthetic_symbols::<A>(s, table_writer, layout)?;
        }
        FileLayout::LinkerScript(_s) => bail!("Wild's PE/COF doesn't support linker scripts yet"),
        FileLayout::NotLoaded | FileLayout::StubLibrary(_) => {}
        FileLayout::Dynamic(_s) => {
            // write_dynamic_file::<A>(s, table_writer, layout)?
        }
    }

    Ok(())
}

struct SymbolTableWriter<'layout, 'out> {
    entries: &'out mut [object::pe::ImageSymbol],
    output_sections: &'layout OutputSections<'layout, Coff>,
    section_layouts: &'layout OutputSectionMap<OutputRecordLayout>,
    strtab_writer: StrTabWriter<'out>,
}

impl<'layout, 'out> SymbolTableWriter<'layout, 'out> {
    fn new(
        start_string_offset: u32,
        buffers: &mut OutputSectionPartMap<&'out mut [u8]>,
        output_sections: &'layout OutputSections<'layout, Coff>,
        section_layouts: &'layout OutputSectionMap<OutputRecordLayout>,
    ) -> Self {
        let entries = slice_from_all_bytes_mut(buffers.take(part_id::SYMTAB));
        let strings = buffers.take(part_id::STRTAB);
        Self {
            entries,
            output_sections,
            section_layouts,
            strtab_writer: StrTabWriter {
                next_offset: start_string_offset,
                out: strings,
            },
        }
    }

    #[inline(always)]
    fn copy_object_symbol(
        &mut self,
        sym: &coff::SymtabEntry,
        sym_index: SymbolIndex,
        symbol_id: SymbolId,
        name: &[u8],
        object: &ObjectLayout<Coff>,
        layout: &CoffLayout,
        value: u64,
        flags: ValueFlags,
    ) -> Result {
        if let Some(section_index) = object.object.symbol_section(sym, sym_index)? {
            self.copy_symbol_with_section(
                sym,
                symbol_id,
                name,
                object,
                layout,
                value,
                flags,
                section_index,
            )?
        } else if sym.is_common() {
            let section_id = /*if sym.st_type() == STT_TLS {
                output_section_id::TBSS
            } else*/ {
                output_section_id::BSS
            };

            Some(self.copy_symbol(sym, name, section_id, value, flags)?)
        } else if sym.is_absolute() {
            self.copy_absolute_symbol(sym, name, flags)
                .with_context(|| {
                    format!("Failed to absolute {}", layout.symbol_debug(symbol_id))
                })?;
            return Ok(());
        } else {
            bail!("Attempted to output a symtab entry with an unexpected section type")
        };

        Ok(())
    }

    #[inline(always)]
    fn copy_symbol_with_section(
        &mut self,
        sym: &coff::SymtabEntry,
        symbol_id: SymbolId,
        name: &[u8],
        object: &ObjectLayout<Coff>,
        layout: &CoffLayout,
        value: u64,
        flags: ValueFlags,
        section_index: object::SectionIndex,
    ) -> Result<Option<&mut object::pe::ImageSymbol>> {
        let section_id = match &object.sections[section_index.0] {
            SectionSlot::Loaded(_)
            | SectionSlot::Sorted(_)
            | SectionSlot::LoadedDebugInfo(_)
            | SectionSlot::MergeStrings(_) => object
                .section_part_id(section_index, &layout.symbol_db.section_part_ids)
                .output_section_id::<Coff>(),
            // SectionSlot::FrameData(..) => output_section_id::EH_FRAME,
            _ => {
                if layout.symbol_db.is_mapping_symbol(symbol_id) {
                    return Ok(None);
                }
                bail!(
                    "Tried to copy a symbol in a section we didn't load. {}",
                    layout.symbol_debug(symbol_id)
                )
            }
        };

        let section_id = layout.output_sections.primary_output_section(section_id);
        Ok(Some(self.copy_symbol(sym, name, section_id, value, flags)?))
    }

    #[inline(always)]
    fn copy_symbol(
        &mut self,
        sym: &coff::SymtabEntry,
        name: &[u8],
        output_section_id: OutputSectionId,
        value: u64,
        flags: ValueFlags,
    ) -> Result<&mut object::pe::ImageSymbol> {
        let shndx = self
            .output_sections
            .output_index_of_section(output_section_id)
            .with_context(|| {
                format!(
                    "internal error: tried to copy symbol `{}` that's in section {} \
                     which is not being output",
                    String::from_utf8_lossy(name),
                    output_section_id,
                )
            })?;
        // Symbol value is section-relatve
        let value = u32::try_from(value - self.section_layouts.get(output_section_id).mem_offset)?;
        self.copy_symbol_shndx(sym, name, shndx as i32, value, flags)
    }

    #[inline(always)]
    fn copy_symbol_shndx(
        &mut self,
        sym: &coff::SymtabEntry,
        name: &[u8],
        shndx: i32,
        value: u32,
        _flags: ValueFlags,
    ) -> Result<&mut object::pe::ImageSymbol> {
        let entry = self.define_symbol(
            // 1-based indexes strike again
            SymbolSection(shndx + 1),
            value,
            name,
        )?;

        entry.typ = sym.raw.typ;
        entry.storage_class = sym.raw.storage_class;
        entry.number_of_aux_symbols = 0;
        Ok(entry)
    }

    #[allow(clippy::unused_self)]
    fn copy_absolute_symbol(
        &self,
        _sym: &coff::SymtabEntry,
        _name: &[u8],
        _flags: ValueFlags,
    ) -> Result<&mut object::pe::ImageSymbol> {
        bail!("Absolute symbols are not supported currently")
    }

    #[inline(always)]
    fn define_symbol(
        &mut self,
        section: SymbolSection,
        value: u32,
        name: &[u8],
    ) -> Result<&mut object::pe::ImageSymbol> {
        let entry = self.entries.split_off_first_mut().with_context(|| {
            format!(
                "Insufficient symbol table allocated for symbol `{}`",
                String::from_utf8_lossy(name),
            )
        })?;
        let mut name_bytes = [0u8; 8];
        if name.len() <= 8 {
            let byte_count = name.len().min(name_bytes.len());
            name_bytes[..byte_count].copy_from_slice(&name[..byte_count]);
        } else {
            let string_offset = self.strtab_writer.write_str(name);
            name_bytes[4..].copy_from_slice(&string_offset.to_le_bytes());
        }

        entry.name = name_bytes;
        entry.value.set(LittleEndian, value);
        entry.section_number.set(LittleEndian, section.0 as u16);

        Ok(entry)
    }

    /// Verifies that we've used up all the space allocated to this writer. i.e. checks that we
    /// didn't allocate too much or missed writing something that we were supposed to write.
    fn check_exhausted(&self) -> Result {
        if !self.entries.is_empty() || !self.strtab_writer.out.is_empty() {
            bail!(
                "Didn't use up all allocated symbol table space. entries={} strings={}",
                self.entries.len(),
                self.strtab_writer.out.len()
            );
        }

        Ok(())
    }
}

fn write_object<'data, A: Arch<Platform = Coff>>(
    object: &ObjectLayout<'data, Coff>,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    table_writer: &mut TableWriter,
    layout: &CoffLayout<'data>,
    trace: &TraceOutput,
) -> Result {
    verbose_timing_phase!("Write object", file_id = object.file_id.as_u32());

    let _span = debug_span!("write_file", filename = %object.input).entered();
    let _file_span = layout.args().common().trace_span_for_file(object.file_id);

    for (i, sec) in object.sections.iter().enumerate() {
        let section_index = object::SectionIndex(i);

        match sec {
            SectionSlot::Loaded(sec) => {
                write_object_section::<A>(object, layout, *sec, section_index, buffers, trace)?;
            }
            SectionSlot::LoadedDebugInfo(_sec) => {
                // write_debug_section::<C, A>(
                //     object,
                //     layout,
                //     *sec,
                //     section_index,
                //     buffers,
                // )?;
            }
            SectionSlot::FrameData(_section_index) => {
                // write_eh_frame_data::<C, A>(
                //     object,
                //     *section_index,
                //     layout,
                //     table_writer,
                //     trace,
                // )?;
            }
            _ => (),
        }
    }

    // for (symbol_id, resolution) in layout.resolutions_in_range(object.symbol_id_range) {
    //     let _span = tracing::trace_span!("Symbol", %symbol_id).entered();
    //     if let Some(res) = resolution {
    //         table_writer
    //             .process_resolution::<A>(Some(layout), layout.args(), res)
    //             .with_context(|| {
    //                 format!(
    //                     "Failed to process `{}` with resolution {res:?}",
    //                     layout.symbol_debug(symbol_id)
    //                 )
    //             })?;
    //
    //         // Dynamic symbols that we define are handled by the epilogue so that they can be
    //         // written in the correct order. Here, we only need to handle weak symbols that we
    //         // reference that aren't defined by any shared objects we're linking against.
    //         if res.flags.is_dynamic() && !layout.args().only_keep_debug() {
    //             let symbol = object
    //                 .object
    //                 .symbol(object.symbol_id_range.id_to_input(symbol_id))?;
    //             let name = object.object.symbol_name(symbol)?;
    //             table_writer.dynsym_writer.copy_symbol_shndx(
    //                 symbol,
    //                 name,
    //                 0,
    //                 0,
    //                 ValueFlags::empty(),
    //             )?;
    //         }
    //     }
    // }

    if !layout.args().should_strip_all() {
        write_symbols(object, &mut table_writer.symbol_table_writer, layout)?;
    }

    Ok(())
}

fn write_object_section<'data, A: Arch<Platform = Coff>>(
    object: &ObjectLayout<'data, Coff>,
    layout: &CoffLayout<'data>,
    section: Section,
    section_index: object::SectionIndex,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    trace: &TraceOutput,
) -> Result {
    // For --only-keep-debug, alloc non-NOTE sections are NOBITS.
    // if layout.args().only_keep_debug() {
    //     let primary_id = layout
    //         .output_sections
    //         .primary_output_section(part_id.output_section_id::<coff::Coff>());
    //     let section_info = layout.output_sections.output_info(primary_id);
    //     let attrs = &section_info.section_attributes;
    //     if crate::only_keep_debug::should_hollow_section(attrs.flags.is_alloc(), attrs.ty) {
    //         return Ok(());
    //     }
    // }

    let out = write_section_raw::<A>(object, layout, section, section_index, buffers)?;

    // We need to reverse the contents and adjust relocations because .ctors/.dtors are executed in
    // reverse order while .init_array/.fini_array are executed in forward order.
    // if should_reverse_contents(
    //     section_index,
    //     part_id,
    //     object.object,
    //     &layout.output_sections,
    // ) {
    //     return write_section_reversed::<C, A>(
    //         object,
    //         layout,
    //         section_index,
    //         table_writer,
    //         trace,
    //         out,
    //     );
    // }

    let relocations = object.relocations(section_index)?;
    apply_relocations::<A, CoffRelocation, _>(
        object,
        out,
        section_index,
        relocations.rel_iter(),
        layout,
        trace,
    )
    .with_context(|| {
        format!(
            "Failed to apply relocations in section `{}` of {}",
            object.object.section_display_name(section_index),
            object.input
        )
    })?;
    Ok(())
}

fn write_section_raw<'out, 'data, A: Arch<Platform = Coff>>(
    object: &ObjectLayout<'data, Coff>,
    layout: &CoffLayout,
    sec: Section,
    section_index: object::SectionIndex,
    buffers: &'out mut OutputSectionPartMap<&mut [u8]>,
) -> Result<&'out mut [u8]> {
    let part_id = object.section_part_id(section_index, &layout.symbol_db.section_part_ids);
    if layout
        .output_sections
        .has_data_in_file(part_id.output_section_id::<Coff>())
    {
        let section_buffer = buffers.get_mut(part_id);
        let allocation_size = sec.capacity(part_id, &layout.output_sections) as usize;
        if section_buffer.len() < allocation_size {
            bail!(
                "Insufficient space allocated to section `{}`. Tried to take {} bytes, but only {} remain",
                object.object.section_display_name(section_index),
                allocation_size,
                section_buffer.len()
            );
        }
        let out = section_buffer.split_off_mut(..allocation_size).unwrap();
        let object_section = object.object.section(section_index)?;
        if object_section.is_no_bits() {
            return Ok(out);
        }

        let section_info = layout
            .output_sections
            .output_info(part_id.output_section_id::<Coff>());
        let section_size = object.object.section_size(object_section)?;
        let section_size = section_size as usize;
        object
            .object
            .copy_section_data(object_section, &mut out[..section_size])?;
        fill_section_padding::<A>(&mut out[section_size..], section_info);

        Ok(&mut out[..section_size])
    } else {
        Ok(&mut [])
    }
}

fn write_symbols<'data>(
    object: &ObjectLayout<'data, Coff>,
    symbol_writer: &mut SymbolTableWriter<'_, '_>,
    layout: &CoffLayout<'data>,
) -> Result {
    for (sym, flags) in object
        .object
        .symbols
        .iter()
        .zip(layout.per_symbol_flags.raw_range(object.symbol_id_range))
        .filter(|(entry, _)| !entry.is_aux)
    {
        let sym_index = sym.index;
        let symbol_id = object.symbol_id_range.input_to_id(sym_index);

        if let Some(info) = SymbolCopyInfo::new(
            object.object,
            sym_index,
            sym,
            symbol_id,
            &layout.symbol_db,
            flags.get(),
            &object.sections,
        ) {
            let Some(res) = layout.local_symbol_resolution(symbol_id) else {
                bail!("Missing resolution for {}", layout.symbol_debug(symbol_id));
            };

            let symbol_value = res.value_for_symbol_table();

            // if sym.st_type() == object::elf::STT_TLS {
            //     symbol_value -= layout.tls_start_address();
            // }

            symbol_writer
                .copy_object_symbol(
                    sym,
                    sym_index,
                    symbol_id,
                    info.name,
                    object,
                    layout,
                    symbol_value,
                    flags.get(),
                )
                .with_context(|| format!("Failed to copy {}", layout.symbol_debug(symbol_id)))?;
        }
    }

    Ok(())
}

fn apply_relocations<
    'data,
    A: Arch<Platform = Coff>,
    R: Relocation<Platform = Coff>,
    I: Iterator<Item = R>,
>(
    object: &ObjectLayout<'data, Coff>,
    out: &mut [u8],
    section_index: object::SectionIndex,
    relocations: I,
    layout: &CoffLayout<'data>,
    trace: &TraceOutput,
) -> Result {
    let section_address = object.section_resolutions[section_index.0]
        .address()
        .context("Attempted to apply relocations to a section that we didn't load")?;

    let mut relocation_count = 0;

    for rel in relocations {
        relocation_count += 1;

        let offset_in_section = rel.offset();

        apply_relocation::<A, R>(
            object,
            offset_in_section,
            &rel,
            section_address,
            layout,
            out,
            trace,
        )
        .with_context(|| {
            format!(
                "Failed to apply {} at offset 0x{offset_in_section:x}",
                display_relocation::<A, R>(object, &rel, layout)
            )
        })?;
    }

    layout
        .relocation_statistics
        .get(
            object
                .section_part_id(section_index, &layout.symbol_db.section_part_ids)
                .output_section_id::<Coff>(),
        )
        .fetch_add(relocation_count, Relaxed);
    Ok(())
}

fn display_relocation<'a, 'data, A: Arch<Platform = Coff>, R: Relocation>(
    object: &'a ObjectLayout<'data, Coff>,
    rel: &'a R,
    layout: &'a CoffLayout<'data>,
) -> DisplayRelocation<'a, 'data, A, R> {
    DisplayRelocation::<'a, 'data, A, R> {
        rel,
        symbol_db: &layout.symbol_db,
        per_symbol_flags: &layout.per_symbol_flags,
        object,
        phantom: PhantomData,
    }
}

struct DisplayRelocation<'a, 'data, A: Arch<Platform = Coff>, R: Relocation> {
    rel: &'a R,
    symbol_db: &'a SymbolDb<'data, Coff>,
    per_symbol_flags: &'a PerSymbolFlags,
    object: &'a ObjectLayout<'data, Coff>,
    phantom: PhantomData<A>,
}

impl<'a, 'data, A: Arch<Platform = Coff>, R: Relocation<Platform = Coff>> Display
    for DisplayRelocation<'a, 'data, A, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "relocation of type {} to ",
            A::rel_type_to_string(self.rel.raw_type())
        )?;
        match self.rel.symbol() {
            None => write!(f, "absolute")?,
            Some(local_symbol_index) => {
                let symbol_id = self.object.symbol_id_range.input_to_id(local_symbol_index);
                write!(
                    f,
                    "{}",
                    self.symbol_db
                        .symbol_debug(self.per_symbol_flags, symbol_id)
                )?;
            }
        }
        Ok(())
    }
}

struct RelocationResolution {
    resolution: Resolution<Coff>,
    symbol_index: SymbolIndex,
    local_symbol_id: SymbolId,
    flags: ValueFlags,
}

#[inline(always)]
fn get_resolution<'data, R: Relocation>(
    rel: &R,
    object_layout: &ObjectLayout<'data, Coff>,
    layout: &CoffLayout<'data>,
) -> Result<RelocationResolution> {
    let symbol_index = rel.symbol().context("Unsupported absolute relocation")?;
    let local_symbol_id = object_layout.symbol_id_range.input_to_id(symbol_index);
    let sym = object_layout.object.symbol(symbol_index)?;
    let section_index = object_layout.object.symbol_section(sym, symbol_index)?;
    let flags = layout.flags_for_symbol(local_symbol_id);
    let resolution = layout
        .symbol_resolution_with_flags(local_symbol_id, flags)
        .or_else(|| {
            section_index.and_then(|section_index| {
                let section_address =
                    object_layout.section_resolutions[section_index.0].address()?;
                let output_offset = platform::Symbol::value(sym);

                Some(Resolution {
                    raw_value: section_address + output_offset,
                    dynamic_symbol_index: None,
                    flags: ValueFlags::empty(),
                    format_specific: Default::default(),
                })
            })
        })
        .with_context(|| {
            format!(
                "Missing resolution for: {}",
                layout.symbol_debug(local_symbol_id)
            )
        })?;
    Ok(RelocationResolution {
        resolution,
        symbol_index,
        local_symbol_id,
        flags,
    })
}

#[inline(always)]
fn apply_relocation<'data, A: Arch<Platform = Coff>, R: Relocation<Platform = Coff>>(
    object_layout: &ObjectLayout<'data, Coff>,
    offset_in_section: u64,
    rel: &R,
    section_address: u64,
    layout: &CoffLayout<'data>,
    out: &mut [u8],
    trace: &TraceOutput,
) -> Result<()> {
    let place = section_address + offset_in_section;
    let _span = tracing::trace_span!(
        "relocation",
        address = place,
        address_hex = %HexU64::new(place)
    )
    .entered();

    let r_type = rel.raw_type();
    let rel_info = A::relocation_from_raw(rel.raw_type())?;
    ensure!(
        rel_info.implicit_addend,
        "PE/COFF relocations always have implicit addend"
    );
    let mut addend = rel.addend() as u64;
    // TODO: Copied from macho and is also similar to `adjust_value_based_on_content`. Figure out
    // if it can be deduplicated
    {
        let RelocationSize::ByteSize(byte_size @ (4 | 8)) = rel_info.size else {
            bail!(
                "Unexpected relocation size for implicit addend read: {}",
                rel_info.size
            );
        };
        let byte_size = usize::from(byte_size);
        let bytes = out
            .get(offset_in_section as usize..)
            .and_then(|bytes| bytes.get(..byte_size))
            .context("implicit addend read extends beyond section data")?;
        let mut buffer = [0u8; 8];
        buffer[..byte_size].copy_from_slice(bytes);
        // addend = addend.wrapping_add(u64_from_slice(&buffer));
        let field = match byte_size {
            4 => i64::from(i32::from_le_bytes(bytes.try_into()?)) as u64,
            8 => u64::from_le_bytes(bytes.try_into()?),
            _ => todo!(),
        };
        addend = addend.wrapping_add(field);
    }

    match rel_info.kind {
        RelocationKind::None => return Ok(()),
        RelocationKind::Relative if rel.symbol().is_none() => {
            if layout.symbol_db.output_kind.is_position_independent() {
                bail!(
                    "relocation of type {} to absolute address cannot be used in \
                    position-independent output; recompile with -fPIC",
                    rel.raw_type()
                );
            }
            let place = section_address + offset_in_section;
            let value = (rel.addend() as u64).wrapping_sub(place);
            rel_info.write_to_buffer(value, &mut out[offset_in_section as usize..])?;
            return Ok(());
        }
        RelocationKind::Absolute
            if layout.symbol_db.output_kind.is_shared_object()
                && A::is_disallowed_in_shared_object(r_type) =>
        {
            bail!(
                "relocation type {} cannot be used when making a shared object; \
                 recompile with -fPIC",
                A::rel_type_to_string(r_type)
            );
        }
        _ => {}
    }
    let RelocationResolution {
        resolution,
        symbol_index,
        local_symbol_id,
        flags,
    } = get_resolution(rel, object_layout, layout)?;
    let reject_dynamic_direct = layout.symbol_db.output_kind.is_position_independent()
        || A::relocation_from_raw(r_type)?.kind == RelocationKind::TpOff;
    if reject_dynamic_direct
        && (flags.is_interposable() || flags.is_dynamic())
        && !flags.needs_copy_relocation()
        && !flags.needs_plt()
        && A::is_disallowed_for_interposable_symbols(r_type)
    {
        bail!(
            "relocation {} cannot be used against symbol; recompile with -fPIC",
            A::rel_type_to_string(r_type)
        );
    }
    let rel_info = A::relocation_from_raw(r_type)?;

    let value = match rel_info.kind {
        RelocationKind::Absolute => resolution.value_with_addend(
            addend as i64,
            symbol_index,
            object_layout,
            &layout.symbol_db.section_part_ids,
            &layout.merged_strings,
            &layout.merged_string_start_addresses,
        )?,
        RelocationKind::Relative => {
            let symbol_plus_addend = if flags.is_undefined_weak() {
                layout.args().image_base().unwrap().wrapping_add(addend)
            } else {
                resolution.value_with_addend(
                    addend as i64,
                    symbol_index,
                    object_layout,
                    &layout.symbol_db.section_part_ids,
                    &layout.merged_strings,
                    &layout.merged_string_start_addresses,
                )?
            };
            // ADDR32NB is RVA, others are RV.
            let base = if r_type == object::pe::IMAGE_REL_AMD64_ADDR32NB {
                layout.args().image_base().unwrap()
            } else {
                place
            };

            symbol_plus_addend.wrapping_sub(base)
        }
        RelocationKind::None => 0,
        _ => {
            unreachable!()
        }
    };

    let offset_in_section = offset_in_section as usize;

    trace.emit(place, || {
        format!(
            "relocation applied flags={flags},\n\
                rel_kind={rel_kind:?},\n\
                value=0x{value:x}, symbol_name={symbol_name}",
            rel_kind = rel_info.kind,
            symbol_name = layout.symbol_db.symbol_name_for_display(local_symbol_id),
        )
    });
    tracing::trace!(
            %flags,
            ?rel_info.kind,
            %rel_info.size,
            value,
            value_hex = %HexU64::new(value),
            symbol_name = %layout.symbol_db.symbol_name_for_display(local_symbol_id),
            "relocation applied");

    rel_info.write_to_buffer(value, &mut out[offset_in_section..])?;

    Ok(())
}

fn write_prelude<A: Arch<Platform = Coff>>(
    prelude: &PreludeLayout<Coff>,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    table_writer: &mut TableWriter<'_, '_>,
    layout: &CoffLayout,
) -> Result {
    verbose_timing_phase!("Write prelude");

    let dos_header: &mut DosHeader = from_bytes_mut(buffers.get_mut(crate::part_id::FILE_HEADER))
        .ok()
        .context("Invalid DOS header allocation")?
        .0;
    populate_dos_header::<A>(layout, dos_header)?;

    let dos_stub: &mut [u8; coff::DOS_STUB.len()] =
        from_bytes_mut(buffers.get_mut(part_id::DOS_STUB))
            .ok()
            .context("Invalid DOS header allocation")?
            .0;
    write_dos_stub::<A>(dos_stub);

    let nt_header: &mut coff::NtHeader = from_bytes_mut(buffers.get_mut(part_id::NT_HEADER))
        .ok()
        .context("Invalid NT header allocation")?
        .0;
    populate_nt_header::<A>(layout, &prelude.header_info, nt_header)?;

    // TODO: object doesn't include this in `coff::NtHeader`, so just zero remaining bytes.
    let remaining_nt_header_bytes =
        &mut buffers.get_mut(part_id::NT_HEADER)[size_of::<coff::NtHeader>()..];
    assert!(!remaining_nt_header_bytes.is_empty());
    remaining_nt_header_bytes.fill(0);

    write_section_headers(table_writer, layout)?;

    // TODO: add helper
    let strtab_size = layout
        .section_layouts
        .get(output_section_id::STRTAB)
        .file_size;
    table_writer
        .symbol_table_writer
        .strtab_writer
        .write_size_header(u32::try_from(strtab_size)?);

    if !layout.args().should_strip_all() {
        write_symbol_table_entries(prelude, &mut table_writer.symbol_table_writer, layout)?;
    }

    write_merged_strings(prelude, buffers, layout);

    Ok(())
}

fn write_merged_strings(
    _prelude: &PreludeLayout<Coff>,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    layout: &CoffLayout,
) {
    // let only_keep_debug = layout.args().only_keep_debug();
    layout.merged_strings.for_each(|section_id, merged| {
        if merged.len() > 0 {
            // TODO
            // let primary_id = layout.output_sections.primary_output_section(section_id);
            // if only_keep_debug
            //     && crate::only_keep_debug::should_hollow_section(
            //         layout.output_sections.section_flags(primary_id).is_alloc(),
            //         layout
            //             .output_sections
            //             .output_info(primary_id)
            //             .section_attributes
            //             .ty,
            //     )
            // {
            //     return;
            // }
            let buffer = buffers.get_mut(section_id.part_id_with_alignment::<Coff>(alignment::MIN));

            write_merged_strings_to_buffer(merged, buffer);
        }
    });

    // TODO
    // if layout.args().should_write_linker_identity {
    //     // Write linker identity into .comment section.
    //     let comment_buffer = buffers.get_mut(
    //         crate::elf::output_section_id::COMMENT.
    // part_id_with_alignment::<Coff>(alignment::MIN),     );
    //     comment_buffer
    //         .split_off_mut(..prelude.identity.len())
    //         .unwrap()
    //         .copy_from_slice(prelude.identity.as_bytes());
    // }
}

pub(crate) fn write_merged_strings_to_buffer(
    merged: &crate::string_merging::MergedStringsSection,
    buffer: &mut &mut [u8],
) {
    merged
        .buckets
        .iter()
        .map(|b| (b, buffer.split_off_mut(..b.len()).unwrap()))
        .par_bridge()
        .for_each(|(bucket, mut buffer)| {
            for string in &bucket.strings {
                let dest = buffer.split_off_mut(..string.len()).unwrap();
                dest.copy_from_slice(string);
            }
        });
}

fn write_symbol_table_entries(
    prelude: &PreludeLayout<Coff>,
    symbol_writer: &mut SymbolTableWriter<'_, '_>,
    layout: &CoffLayout,
) -> Result {
    let internal_symbols = &prelude.internal_symbols;

    write_internal_symbols(internal_symbols, layout, symbol_writer)?;
    Ok(())
}

fn write_synthetic_symbols<'data, A: Arch<Platform = Coff>>(
    syn: &SyntheticSymbolsLayout<Coff>,
    table_writer: &mut TableWriter<'_, '_>,
    layout: &CoffLayout<'data>,
) -> Result {
    verbose_timing_phase!("Write synthetic symbols");

    if !layout.args().should_strip_all() {
        write_internal_symbols(
            &syn.internal_symbols,
            layout,
            &mut table_writer.symbol_table_writer,
        )?;
    }

    Ok(())
}

fn write_epilogue<A: Arch<Platform = Coff>>(
    _epilogue: &EpilogueLayout<Coff>,
    _buffers: &mut OutputSectionPartMap<&mut [u8]>,
    _table_writer: &mut TableWriter<'_, '_>,
    _layout: &CoffLayout,
    _trace: &TraceOutput,
) {
    verbose_timing_phase!("Write epilogue");

    // TODO: LLD has .buildid section header
    // The actual build-id will be filled in later once all writing has completed. It's important
    // that we fill it with zeros now however, since if we're overwriting an existing file, there
    // might be other data there and we don't zero it, then the build ID will be hashing that data.
    // let build_id_buffer = buffers.get_mut(part_id::NOTE_GNU_BUILD_ID);
    // build_id_buffer.fill(0);

    // for sorted_section in &layout.script_sorted_sections {
    //     let FileLayout::Object(object) = layout.file_layout(sorted_section.file_id) else {
    //         unreachable!();
    //     };
    //
    //     if let SectionSlot::Sorted(sec) =
    // &object.sections[sorted_section.section_index.as_usize()]     {
    //         write_object_section::<A>(
    //             object,
    //             layout,
    //             sec.section,
    //             sorted_section.section_index.into(),
    //             buffers,
    //             trace,
    //         )?;
    //     }
    // }
}

fn write_internal_symbols(
    internal_symbols: &InternalSymbols<Coff>,
    layout: &CoffLayout,
    symbol_writer: &mut SymbolTableWriter<'_, '_>,
) -> Result {
    for (local_index, def_info) in internal_symbols.symbol_definitions.iter().enumerate() {
        if def_info.name.is_empty() {
            continue;
        }
        let symbol_id = internal_symbols.start_symbol_id.add_usize(local_index);
        if !layout.symbol_db.is_canonical(symbol_id) {
            continue;
        }
        let Some(resolution) = layout.local_symbol_resolution(symbol_id) else {
            continue;
        };

        let symbol_name = layout.symbol_db.symbol_name(symbol_id)?;

        // For Redirect, get attributes from the target symbol
        let (mut shndx, st_type) = if matches!(
            def_info.placement,
            crate::parsing::SymbolPlacement::Redirect(_)
        ) {
            unreachable!();
        } else {
            let shndx = def_info
                .section_id()
                .map(|section_id| {
                    let section_id = layout.output_sections.primary_output_section(section_id);

                    layout
                        .output_sections
                        .output_index_of_section(section_id)
                        .with_context(|| {
                            format!(
                                "symbol '{}' in section '{}' that we're not going to output {resolution:?}",
                                layout.symbol_db.symbol_name_for_display(symbol_id),
                                layout.output_sections.display_name(section_id)
                            )
                        })
                })
                .transpose()?
                .map_or(object::pe::IMAGE_SYM_ABSOLUTE, |i| SymbolSection(i as i32));

            (shndx, def_info.symbol.raw.typ)
        };

        // Move symbols that are in our header (section 0) into the first section, otherwise they'll
        // show up as undefined.
        // if matches!(shndx, SymbolSection::Index(0)) {
        //     shndx = SymbolSection::Index(1);
        // }
        // 1-based offsets
        if shndx.0 >= 0 {
            shndx = SymbolSection(shndx.0 + 1);
        }

        // TODO: section_id is also called above
        let address = u32::try_from(
            resolution.value()
                - symbol_writer
                    .section_layouts
                    .get(def_info.section_id().unwrap())
                    .mem_offset,
        )?;

        // if platform::Symbol::is_tls(&def_info.symbol) {
        //     address -= layout.tls_start_address();
        // }

        let entry = symbol_writer
            .define_symbol(shndx, address, symbol_name.bytes())
            .with_context(|| format!("Failed to write {}", layout.symbol_debug(symbol_id)))?;

        entry.typ = st_type;
        entry.storage_class = def_info.symbol.raw.storage_class;
        entry.number_of_aux_symbols = 0;
    }
    Ok(())
}

fn write_section_headers(table_writer: &mut TableWriter<'_, '_>, layout: &CoffLayout) -> Result {
    let output_sections = &layout.output_sections;

    let mut order = layout.output_order.into_iter().peekable();

    while let Some(event) = order.next() {
        let OrderEvent::Section(section_id) = event else {
            continue;
        };

        let output_info = output_sections.output_info(section_id);
        let section_layout = layout.merged_section_layouts.get(section_id);

        if output_sections
            .output_index_of_section(section_id)
            .is_none()
        {
            continue;
        }

        if output_info.section_attributes.is_null {
            continue;
        }

        while let Some(OrderEvent::Section(next_section_id)) = order.peek()
            && let Some(primary_id) = output_sections.merge_target(*next_section_id)
        {
            debug_assert_bail!(
                primary_id == section_id,
                "Section order mismatch {} != {}",
                output_sections.section_debug(primary_id),
                output_sections.section_debug(section_id),
            );
            order.next();
        }

        let name = layout.output_sections.name(section_id).with_context(|| {
            format!(
                "Missing name for section {}",
                layout.output_sections.section_debug(section_id)
            )
        })?;
        let mut name_bytes = [0u8; 8];
        if name.len() > 8
            && layout
                .output_sections
                .section_flags(section_id)
                .contains(object::pe::IMAGE_SCN_MEM_DISCARDABLE)
        {
            // table_writer.write_section_header_string(name.bytes())?;
            eprintln!(
                "long section names are not supported yet, section name for `{name}` will be truncated"
            );
        }
        let byte_count = name.len().min(name_bytes.len());
        name_bytes[..byte_count].copy_from_slice(&name.bytes()[..byte_count]);

        let entry = table_writer.take_section_header()?;
        entry.name = name_bytes;
        entry
            .virtual_size
            .set(LittleEndian, section_layout.mem_size.try_into()?);
        entry.virtual_address.set(
            LittleEndian,
            rva(section_layout.mem_offset, layout).try_into()?,
        );
        entry.size_of_raw_data.set(
            LittleEndian,
            section_layout
                .alignment
                .align_up(section_layout.file_size as u64)
                .try_into()?,
        );
        entry
            .pointer_to_raw_data
            .set(LittleEndian, section_layout.file_offset.try_into()?);
        entry
            .pointer_to_relocations
            .set(LittleEndian, Default::default());
        entry
            .pointer_to_linenumbers
            .set(LittleEndian, Default::default());
        entry
            .number_of_relocations
            .set(LittleEndian, Default::default());
        entry
            .number_of_linenumbers
            .set(LittleEndian, Default::default());
        entry.characteristics.set(
            LittleEndian,
            layout
                .output_sections
                .section_flags(section_id)
                .without(object::pe::SectionFlags(object::pe::IMAGE_SCN_ALIGN_MASK)),
        );
    }
    Ok(())
}

struct StrTabWriter<'out> {
    next_offset: u32,
    out: &'out mut [u8],
}

impl StrTabWriter<'_> {
    /// Writes a string to the string table. Returns the offset within the string table at which the
    /// string was written.
    fn write_str(&mut self, str: &[u8]) -> u32 {
        let len_with_terminator = str.len() + 1;
        let lib_name_out = self.out.split_off_mut(..len_with_terminator).unwrap();
        lib_name_out[..str.len()].copy_from_slice(str);
        lib_name_out[str.len()] = 0;
        let offset = self.next_offset;
        self.next_offset += len_with_terminator as u32;
        offset
    }

    fn write_size_header(&mut self, total_size: u32) {
        let header = self.out.split_off_mut(..4).unwrap();
        header.copy_from_slice(&total_size.to_le_bytes());
        self.next_offset += 4;
    }
}

fn fill_section_padding<A: Arch<Platform = Coff>>(
    padding: &mut [u8],
    section_info: &SectionOutputInfo<Coff>,
) {
    if let Some(pattern) = section_info.fill {
        let chunks = padding.chunks_mut(4);
        for chunk in chunks {
            let len = chunk.len();
            chunk.copy_from_slice(&pattern[..len]);
        }
    } else {
        // TODO: fill_section_padding uses ELF flags
        // A::fill_section_padding(padding, section_info.section_attributes.flags);
        padding.fill(0);
    }
}

fn rva(address: u64, layout: &CoffLayout) -> u64 {
    address - layout.args().image_base().unwrap()
}
