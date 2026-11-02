#![allow(dead_code, unused_variables)]

use crate::OutputFileData;
use crate::bail;
use crate::coff;
use crate::coff::Coff;
use crate::coff::CoffRelocation;
use crate::coff::DosHeader;
use crate::coff::SectionHeader;
use crate::coff::output_section_id;
use crate::coff::part_id;
use crate::error::Context;
use crate::error::Result;
use crate::file_writer::SizedOutput;
use crate::file_writer::split_buffers_by_alignment;
use crate::file_writer::split_output_by_group;
use crate::file_writer::split_output_into_sections;
use crate::layout::FileLayout;
use crate::layout::HeaderInfo;
use crate::layout::Layout;
use crate::layout::ObjectLayout;
use crate::layout::PartialLinkSingleton;
use crate::layout::PreludeLayout;
use crate::layout::Resolution;
use crate::layout::Section;
use crate::output_kind::OutputKind;
use crate::output_section_id::SectionOutputInfo;
use crate::output_section_map::OutputSectionMap;
use crate::output_section_part_map::OutputSectionPartMap;
use crate::output_trace::HexU64;
use crate::output_trace::TraceOutput;
use crate::part_id::PartId;
use crate::platform;
use crate::platform::Arch;
use crate::platform::Args;
use crate::platform::ObjectFile;
use crate::platform::Platform;
use crate::platform::Relocation;
use crate::platform::RelocationSequence;
use crate::platform::SectionHeader as _;
use crate::resolution::SectionSlot;
use crate::symbol_db::SymbolDb;
use crate::symbol_db::SymbolId;
use crate::timing::timing_phase;
use crate::timing::verbose_timing_phase;
use crate::value_flags::PerSymbolFlags;
use crate::value_flags::ValueFlags;
use hashbrown::HashMap;
use linker_utils::elf::RelocationKind;
use linker_utils::elf::RelocationSize;
use linker_utils::relaxation::opt_input_to_output;
use object::LittleEndian;
use object::SymbolIndex;
use object::from_bytes_mut;
use object::pe::ImageFileHeader;
use object::pe::ImageOptionalHeader64;
use object::slice_from_bytes_mut;
use object::write::WritableBuffer;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::ParallelIterator as _;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::atomic::Ordering::Relaxed;
use tracing::debug_span;

type CoffLayout<'data> = Layout<'data, coff::Coff>;

pub(crate) fn write<'data, A: Arch<Platform = coff::Coff>>(
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

fn write_file_contents<'data, A: Arch<Platform = coff::Coff>>(
    sized_output: &mut SizedOutput<impl OutputFileData>,
    layout: &CoffLayout<'data>,
) -> Result {
    timing_phase!("Write data to file");
    let (mut section_buffers, padding) = split_output_into_sections(layout, &mut sized_output.out);

    fill_padding_for_sections::<A>(layout, padding);

    let sym_index_map = Vec::new();

    let mut writable_buckets = split_buffers_by_alignment(&mut section_buffers, layout);
    let groups_and_buffers = split_output_by_group(layout, &mut writable_buckets);
    groups_and_buffers
        .into_par_iter()
        .with_max_len(1)
        .try_for_each(|(group, mut buffers)| -> Result {
            verbose_timing_phase!("Write group");

            let mut table_writer = TableWriter::from_layout(layout, &mut buffers);

            for file in &group.files {
                write_file::<A>(
                    file,
                    &mut buffers,
                    &mut table_writer,
                    layout,
                    &sized_output.trace,
                    &sym_index_map,
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
    output_kind: OutputKind,
    tls: Range<u64>,
    _phantom: std::marker::PhantomData<&'layout CoffLayout<'out>>,
}

impl<'layout, 'out> TableWriter<'layout, 'out> {
    fn from_layout(
        layout: &'layout CoffLayout,
        buffers: &mut OutputSectionPartMap<&'out mut [u8]>,
    ) -> TableWriter<'layout, 'out> {
        Self::new(
            layout.symbol_db.output_kind,
            layout.tls_start_address()..layout.tls_end_address(),
            buffers,
        )
    }

    fn new(
        output_kind: OutputKind,
        tls: Range<u64>,
        _buffers: &mut OutputSectionPartMap<&'out mut [u8]>,
    ) -> TableWriter<'layout, 'out> {
        TableWriter {
            output_kind,
            tls,
            _phantom: Default::default(),
        }
    }

    /// Checks that we used all of the entries that we requested during layout.
    fn validate_empty(&self, mem_sizes: &OutputSectionPartMap<u64>) -> crate::error::Result {
        Ok(())
    }
}

fn fill_padding_for_sections<A: Arch<Platform = coff::Coff>>(
    layout: &Layout<'_, coff::Coff>,
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

fn populate_dos_header<A: Arch<Platform = coff::Coff>>(
    layout: &CoffLayout,
    header: &mut coff::DosHeader,
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

fn write_dos_stub<A: Arch<Platform = coff::Coff>>(layout: &CoffLayout, mut bytes: &mut [u8]) {
    bytes.write_bytes(coff::DOS_STUB);
}

fn populate_nt_header<A: Arch<Platform = coff::Coff>>(
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

    let e = LittleEndian;
    let first_regular_section = layout.section_layouts.get(output_section_id::TEXT);

    let file_header = ImageFileHeader {
        machine: object::pe::IMAGE_FILE_MACHINE_AMD64.into(),
        number_of_sections: u16::try_from(header_info.num_output_sections_with_content)?.into(),
        time_date_stamp: Default::default(),
        pointer_to_symbol_table: Default::default(),
        number_of_symbols: Default::default(),
        size_of_optional_header: coff::SIZE_OF_OPTIONAL_HEADER64.into(),
        characteristics: characteristics.into(),
    };
    // TODO: Skips a lot of logic compared to ELF one
    let entry_point_address = u32::try_from(rva(
        layout.resolved_entry_symbol_address()?.unwrap(),
        layout,
    ))?;
    let optional_header = ImageOptionalHeader64 {
        magic: object::pe::IMAGE_NT_OPTIONAL_HDR64_MAGIC.into(),
        major_linker_version: 14,
        minor_linker_version: 0,
        size_of_code: Default::default(),
        // TODO: filter only sections with CODE characteristics
        size_of_initialized_data: u32::try_from(
            layout
                .section_layouts
                .iter()
                .map(|(_, sec)| sec.file_size)
                .sum::<usize>(),
        )?
        .into(),
        size_of_uninitialized_data: u32::try_from(
            layout
                .section_layouts
                .get(output_section_id::TEXT)
                .file_size,
        )?
        .into(),
        address_of_entry_point: entry_point_address.into(),
        base_of_code: u32::try_from(rva(first_regular_section.mem_offset, layout))?.into(),
        image_base: layout.args().image_base().unwrap().into(),
        section_alignment: u32::try_from(first_regular_section.alignment.value())?.into(),
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

fn populate_section_headers<A: Arch<Platform = coff::Coff>>(
    layout: &CoffLayout,
    header_info: &HeaderInfo,
    headers: &mut [SectionHeader],
) -> Result {
    let e = LittleEndian;

    let mut sections_iter = layout.section_layouts.iter().skip_while(|(idx, _)| {
        layout
            .output_sections
            .output_info(*idx)
            .section_attributes
            .is_null
    });
    for header in headers {
        let (sec_idx, sec) = sections_iter.next().unwrap();
        let sec_name = layout.output_sections.name(sec_idx).unwrap();
        // debug_assert!(
        //     sec_name.len() <= 8,
        //     "long section names are not supported yet"
        // );
        let mut name = [0u8; 8];
        for (name_idx, byte) in sec_name.0.iter().enumerate() {
            // TODO: hack
            if name_idx < 8 {
                name[name_idx] = *byte;
            }
        }
        *header = SectionHeader {
            name,
            virtual_size: u32::try_from(sec.mem_size)?.into(),
            virtual_address: u32::try_from(rva(sec.mem_offset, layout))?.into(),
            size_of_raw_data: u32::try_from(sec.alignment.align_up(sec.file_size as u64))?.into(),
            pointer_to_raw_data: u32::try_from(sec.file_offset)?.into(),
            pointer_to_relocations: Default::default(),
            pointer_to_linenumbers: Default::default(),
            number_of_relocations: Default::default(),
            number_of_linenumbers: Default::default(),
            characteristics: layout.output_sections.section_flags(sec_idx).into(),
        };
    }

    Ok(())
}

fn write_file<'data, A: Arch<Platform = coff::Coff>>(
    file: &FileLayout<'data, coff::Coff>,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    table_writer: &mut TableWriter<'_, '_>,
    layout: &CoffLayout<'data>,
    trace: &TraceOutput,
    sym_index_map: &[Option<u32>],
) -> Result {
    match file {
        FileLayout::Object(s) => {
            write_object::<A>(s, buffers, table_writer, layout, trace, sym_index_map)?;
        }
        FileLayout::Prelude(s) => {
            write_prelude::<A>(s, buffers, layout, sym_index_map)?;
        }
        FileLayout::Epilogue(s) => {
            // write_epilogue::<A>(s, buffers, table_writer, layout, trace)?
        }
        FileLayout::SyntheticSymbols(s) => {
            // write_synthetic_symbols::<A>(s, table_writer, layout)?;
        }
        FileLayout::LinkerScript(s) => todo!("Wild's coff doesn't support linker scripts yet"),
        FileLayout::NotLoaded | FileLayout::StubLibrary(_) => {}
        FileLayout::Dynamic(s) => {
            // write_dynamic_file::<A>(s, table_writer, layout)?
        }
    }

    Ok(())
}

fn write_object<'data, A: Arch<Platform = coff::Coff>>(
    object: &ObjectLayout<'data, coff::Coff>,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    table_writer: &mut TableWriter<'_, '_>,
    layout: &CoffLayout<'data>,
    trace: &TraceOutput,
    sym_index_map: &[Option<u32>],
) -> Result {
    verbose_timing_phase!("Write object", file_id = object.file_id.as_u32());

    let _span = debug_span!("write_file", filename = %object.input).entered();
    let _file_span = layout.args().common().trace_span_for_file(object.file_id);

    for (i, sec) in object.sections.iter().enumerate() {
        let section_index = object::SectionIndex(i);

        match sec {
            SectionSlot::Loaded(sec)
            | SectionSlot::PartialLinkSingleton(PartialLinkSingleton { section: sec, .. }) => {
                let part_id =
                    object.section_part_id(section_index, &layout.symbol_db.section_part_ids);
                let input_header = object.object.section(section_index)?;

                write_object_section::<A>(
                    object,
                    layout,
                    *sec,
                    section_index,
                    buffers,
                    table_writer,
                    trace,
                )?;
            }
            SectionSlot::LoadedDebugInfo(sec) => {
                // write_debug_section::<C, A>(
                //     object,
                //     layout,
                //     *sec,
                //     section_index,
                //     buffers,
                // )?;
            }
            SectionSlot::FrameData(section_index) => {
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

    // if !layout.args().should_strip_all() || layout.args().should_output_partial_object() {
    //     write_symbols(object, &mut table_writer.debug_symbol_writer, layout)?;
    // }

    Ok(())
}

fn write_object_section<'data, A: Arch<Platform = coff::Coff>>(
    object: &ObjectLayout<'data, coff::Coff>,
    layout: &CoffLayout<'data>,
    section: Section,
    section_index: object::SectionIndex,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    table_writer: &mut TableWriter<'_, '_>,
    trace: &TraceOutput,
) -> Result {
    let part_id = object.section_part_id(section_index, &layout.symbol_db.section_part_ids);

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

    let (out, instruction_mapping) =
        write_section_raw::<A>(object, layout, section, section_index, buffers)?;

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
        &instruction_mapping,
        layout,
        table_writer,
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

fn write_section_raw<'out, 'data, A: Arch<Platform = coff::Coff>>(
    object: &ObjectLayout<'data, coff::Coff>,
    layout: &CoffLayout,
    sec: Section,
    section_index: object::SectionIndex,
    buffers: &'out mut OutputSectionPartMap<&mut [u8]>,
) -> Result<(&'out mut [u8], HashMap<u64, u64>)> {
    let part_id = object.section_part_id(section_index, &layout.symbol_db.section_part_ids);
    if layout
        .output_sections
        .has_data_in_file(part_id.output_section_id::<coff::Coff>())
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
            return Ok((out, HashMap::new()));
        }

        let section_info = layout
            .output_sections
            .output_info(part_id.output_section_id::<coff::Coff>());
        let section_size = object.object.section_size(object_section)?;
        let section_size = section_size as usize;
        object
            .object
            .copy_section_data(object_section, &mut out[..section_size])?;
        fill_section_padding::<A>(&mut out[section_size..], section_info);

        Ok((&mut out[..section_size], HashMap::new()))
    } else {
        Ok((&mut [], HashMap::new()))
    }
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
    mut relocations: I,
    instruction_mapping: &HashMap<u64, u64>,
    layout: &CoffLayout<'data>,
    table_writer: &mut TableWriter<'_, '_>,
    trace: &TraceOutput,
) -> Result {
    let section_address = object.section_resolutions[section_index.0]
        .address()
        .context("Attempted to apply relocations to a section that we didn't load")?;
    let object_section = object.object.section(section_index)?;
    // let section_flags = object_section.sh_flags(LittleEndian);
    let section_flags = object_section.characteristics.get(LittleEndian);
    // let mut modifier = RelocationModifier::Normal;
    let section_info = SectionInfo {
        section_address,
        is_writable: object_section.is_writable(),
        section_flags,
        part_id: object.section_part_id(section_index, &layout.symbol_db.section_part_ids),
    };

    let mut relocation_count = 0;
    // let mut relocation_cache = RelocationCache::<R>::default();
    let relax_deltas = object.section_relax_deltas.get(section_index.0);
    let mut relax_cursor = relax_deltas.map(|deltas| deltas.cursor());

    while let Some(rel) = relocations.next() {
        relocation_count += 1;
        // if A::high_part_relocations().contains(&rel.raw_type()) {
        // let cache_offset = opt_input_to_output(relax_deltas, rel.offset());
        // relocation_cache.high_part_symbols.insert(cache_offset, rel);
        // }

        // if modifier == RelocationModifier::SkipNextRelocation {
        //     modifier = RelocationModifier::Normal;
        //     relocation_cache.previous = Some(rel);
        //     continue;
        // }

        // When relaxation deltas are present, translate the relocation's input
        // offset to the corresponding output offset so that it points to the
        // correct position in the (compacted) output buffer.
        let offset_in_section = match relax_cursor.as_mut() {
            Some(cursor) => cursor.translate(rel.offset()),
            None => rel.offset(),
        };

        // modifier =
        apply_relocation::<A, R, _>(
            object,
            offset_in_section,
            &rel,
            section_info,
            layout,
            out,
            table_writer,
            trace,
            // &relocation_cache,
            &relocations,
            // relax_deltas,
            instruction_mapping,
        )
        .with_context(|| {
            format!(
                "Failed to apply {} at offset 0x{offset_in_section:x}",
                display_relocation::<A, R>(object, &rel, layout)
            )
        })?;
        // relocation_cache.previous = Some(rel);
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

#[derive(Clone, Copy)]
struct SectionInfo<S: platform::SectionFlags> {
    section_address: u64,
    is_writable: bool,
    section_flags: S,
    part_id: PartId,
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
                let output_offset = opt_input_to_output(
                    object_layout.section_relax_deltas.get(section_index.0),
                    crate::platform::Symbol::value(sym),
                );

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
fn apply_relocation<
    'data,
    A: Arch<Platform = Coff>,
    R: Relocation<Platform = Coff>,
    I: Iterator<Item = R>,
>(
    object_layout: &ObjectLayout<'data, Coff>,
    mut offset_in_section: u64,
    rel: &R,
    section_info: SectionInfo<object::pe::SectionFlags>,
    layout: &CoffLayout<'data>,
    out: &mut [u8],
    table_writer: &mut TableWriter<'_, '_>,
    trace: &TraceOutput,
    // relocation_cache: &RelocationCache<R>,
    relocation_iterator: &I,
    // relax_deltas: Option<&SectionRelaxDeltas>,
    instruction_mapping: &HashMap<u64, u64>,
    // ) -> Result<RelocationModifier> {
) -> Result<()> {
    let section_address = section_info.section_address;
    // Erratum instructions are moved to the veneer location after a section.
    if let Some(&new_place) = instruction_mapping.get(&(section_address + offset_in_section)) {
        offset_in_section = new_place - section_address;
    }
    let original_place = section_address + offset_in_section;
    let _span = tracing::trace_span!(
        "relocation",
        address = original_place,
        address_hex = %HexU64::new(original_place)
    )
    .entered();

    let r_type = rel.raw_type();
    let mut addend = rel.addend();

    match A::relocation_from_raw(r_type)?.kind {
        // RelocationKind::None => return Ok(RelocationModifier::Normal),
        RelocationKind::None => return Ok(()),
        // RelocationKind::Alignment => {
        //     let addend = addend as u64;
        //     let removed_bytes =
        //         relax_deltas.map_or(0u64, |d| u64::from(d.delta_bytes_at(rel.offset())));
        //     let padding_bytes = addend.saturating_sub(removed_bytes) as usize;
        //     let offset_in_section = offset_in_section as usize;
        //     A::fill_nop_padding(&mut out[offset_in_section..offset_in_section + padding_bytes]);
        //
        //     return Ok(RelocationModifier::Normal);
        // }
        RelocationKind::Relative if rel.symbol().is_none() => {
            if layout.symbol_db.output_kind.is_position_independent() {
                bail!(
                    "relocation of type {} to absolute address cannot be used in \
                    position-independent output; recompile with -fPIC",
                    rel.raw_type()
                );
            }
            let place = section_info.section_address + offset_in_section;
            let value = (rel.addend() as u64).wrapping_sub(place);
            A::relocation_from_raw(r_type)?
                .write_to_buffer(value, &mut out[offset_in_section as usize..])?;
            // return Ok(RelocationModifier::Normal);
            return Ok(());
        }
        RelocationKind::Absolute | RelocationKind::TpOff
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
    // let mut next_modifier = RelocationModifier::Normal;
    let rel_info;
    let output_kind = layout.symbol_db.output_kind;

    // let relaxation = A::new_relaxation(
    //     r_type,
    //     out,
    //     offset_in_section,
    //     flags,
    //     output_kind,
    //     section_info.section_flags,
    //     relax_deltas,
    //     resolution.raw_value,
    //     section_address,
    //     rel.addend(),
    //     relocation_cache
    //         .previous
    //         .as_ref()
    //         .filter(|r| r.symbol() == rel.symbol())
    //         .map(|r| PreviousRelocationInfo {
    //             kind: r.raw_type(),
    //             offset: r.offset(),
    //             symbol: r.symbol(),
    //             addend: r.addend(),
    //         }),
    // )
    // .filter(|relaxation| layout.args().relax || relaxation.is_mandatory());
    //
    // if let Some(relaxation) = &relaxation {
    //     rel_info = relaxation.rel_info();
    //     relaxation.apply(out, &mut offset_in_section, &mut addend);
    //     next_modifier = relaxation.next_modifier();
    // } else {
    rel_info = A::relocation_from_raw(r_type)?;
    // }

    // Compute place to which IP-relative relocations will be relative. This is different to
    // `original_place` in that our `offset_in_section` may have been adjusted by a relaxation.
    let place = section_address + offset_in_section;

    let undefined_weak_target = if flags.is_undefined_weak() && !flags.needs_plt() {
        A::undefined_weak_target(r_type, place)
    } else {
        None
    };
    // let mask = get_page_mask(rel_info.mask);
    // let bias = rel_info.bias.map_or(0, Bias::value);
    // Set when a ppc64 branch targets a PLT stub. The stub's first instruction saves r2 and the nop
    // after `bl` has to load it back.
    // let mut restore_caller_toc = false;
    let mut value = match rel_info.kind {
        RelocationKind::Absolute => write_absolute_relocation::<A>(
            table_writer,
            resolution,
            rel_info.size,
            place,
            addend,
            section_info,
            symbol_index,
            object_layout,
            layout,
        )?,
        /*
        RelocationKind::SymbolSize => {
            if section_info.section_flags.is_alloc()
                && resolution.flags.is_interposable()
                && (resolution.flags.is_dynamic()
                    || layout.symbol_db.output_kind.is_shared_object())
            {
                table_writer.write_rela_dyn_general(
                    place,
                    resolution.dynamic_symbol_index()?,
                    r_type,
                    addend,
                )?;

                0
            } else {
                canonical_symbol_size(layout, local_symbol_id)?.wrapping_add(addend as u64)
            }
        }
                RelocationKind::AbsoluteSet
                | RelocationKind::AbsoluteSetWord6
                | RelocationKind::AbsoluteAddition
                | RelocationKind::AbsoluteAdditionWord6
                | RelocationKind::AbsoluteSubtraction
                | RelocationKind::AbsoluteSubtractionWord6 => resolution.value_with_addend(
                    addend,
                    symbol_index,
                    object_layout,
                    &layout.symbol_db.section_part_ids,
                    &layout.merged_strings,
                    &layout.merged_string_start_addresses,
                )?,
        */
        /*        RelocationKind::AbsoluteLowPart => resolution
                            .value_with_addend(
                                addend,
                                symbol_index,
                                object_layout,
                                &layout.symbol_db.section_part_ids,
                                &layout.merged_strings,
                                &layout.merged_string_start_addresses,
                            )?
                            .bitand(mask.symbol_plus_addend),
                */
        RelocationKind::Relative => {
            let mut symbol_plus_addend = if let Some(target) = undefined_weak_target {
                target.wrapping_add(addend as u64)
            } else {
                resolution.value_with_addend(
                    addend,
                    symbol_index,
                    object_layout,
                    &layout.symbol_db.section_part_ids,
                    &layout.merged_strings,
                    &layout.merged_string_start_addresses,
                )?
            };
            // A function defined in this object resolves to its global entry point. An
            // interposable branch lands on the PLT stub instead, so the stub can load the
            // JMP_SLOT into r12. A hidden reference stays on the local entry point. GCC omits
            // the nop after `bl` when the call is expected to stay in this module. Those sites
            // keep the local entry. A tail `b` does not return, so it can still use the stub.
            // if rel_info.size.is_ppc64_branch()
            //     && flags.needs_plt()
            //     && flags.is_interposable()
            //     && flags.has_link_time_address()
            //     && let Some(plt) = resolution.format_specific.plt_address
            // {
            //     let branch_offset = offset_in_section as usize;
            //     let call_can_use_plt_stub =
            //         out.get(branch_offset..branch_offset + 4)
            //             .is_some_and(|branch| {
            //                 let insn = u32::from_le_bytes(branch.try_into().unwrap());
            //                 // LK is the low bit. A tail branch has no TOC-restore slot.
            //                 insn & 1 != 1
            //                     || out.get(branch_offset + 4..branch_offset + 8).is_some_and(
            //                         |next| {
            //                             u32::from_le_bytes(next.try_into().unwrap()) ==
            // 0x6000_0000                         },
            //                     )
            //             });
            //     if call_can_use_plt_stub {
            //         symbol_plus_addend = plt.get().wrapping_add(addend as u64);
            //     }
            // }

            // let targets_plt_stub = flags.needs_plt()
            //     && resolution
            //         .format_specific
            //         .plt_address
            //         .is_some_and(|plt| symbol_plus_addend == plt.get().wrapping_add(addend as
            // u64)); let local_entry = if rel_info.size.is_ppc64_branch() && !targets_plt_stub {
            //     A::local_entry_offset(callee_st_other(layout, local_symbol_id))
            // } else {
            //     0
            // };
            // restore_caller_toc = targets_plt_stub && rel_info.size.is_ppc64_branch();
            let local_entry = 0;

            symbol_plus_addend
                .wrapping_add(local_entry)
                // .wrapping_add(bias)
                // .bitand(mask.symbol_plus_addend)
                .wrapping_sub(
                    place, // .bitand(mask.place)
                )
        }
        /*
                RelocationKind::RelativeLoongArchHigh => highest_relocation_with_bias(
                    resolution.value_with_addend(
                        addend,
                        symbol_index,
                        object_layout,
                        &layout.symbol_db.section_part_ids,
                        &layout.merged_strings,
                        &layout.merged_string_start_addresses,
                    )?,
                    place,
                ),
                RelocationKind::RelativeRiscVLow12 => {
                    // The iterator is used for e.g. R_RISCV_PCREL_HI20 & R_RISCV_PCREL_LO12_I pair of
                    // relocations where the later one actually points to a label of the HI20
                    // relocations and thus we need to find it. The relocation is typically
                    // right before the LO12_* relocation.
                    ensure!(
                        addend == 0,
                        "Unexpected addend for R_RISCV_PCREL_LO12 relocation"
                    );
                    let hi_offset_in_section = resolution
                        .value_with_addend(
                            addend,
                            symbol_index,
                            object_layout,
                            &layout.symbol_db.section_part_ids,
                            &layout.merged_strings,
                            &layout.merged_string_start_addresses,
                        )?
                        .wrapping_sub(section_address);
                    let hi_rel = relocation_cache
                        .high_part_symbols
                        .get(&hi_offset_in_section)
                        .copied()
                        .or_else(|| {
                            // It's very unlikely that a high part follows the low part:
                            relocation_iterator.clone().find_map(|r| {
                                if let Ok(r) = r
                                    && A::high_part_relocations().contains(&r.raw_type())
                                {
                                    let r_output_offset = opt_input_to_output(relax_deltas, r.offset());
                                    if r_output_offset == hi_offset_in_section {
                                        return Some(r);
                                    }
                                }
                                None
                            })
                        })
                        .context("Missing High relocation connected with R_RISCV_PCREL_LO12")?;

                    let hi_rel_info = A::relocation_from_raw(hi_rel.raw_type())?;
                    let addend = hi_rel.addend();
                    let crate::elf_writer::RelocationResolution {
                        resolution,
                        symbol_index,
                        ..
                    } = crate::elf_writer::get_resolution(&hi_rel, object_layout, layout).with_context(
                        || "Missing High resolution connected to R_RISCV_PCREL_LO12".to_string(),
                    )?;
                    let place = section_address + hi_offset_in_section;

                    // Only a subset of relocations is referenced by R_RISCV_PCREL_LO12 relocations.
                    match hi_rel_info.kind {
                        RelocationKind::Relative => resolution
                            .value_with_addend(
                                addend,
                                symbol_index,
                                object_layout,
                                &layout.symbol_db.section_part_ids
                                    [object_layout.section_id_range.as_usize()],
                                &layout.merged_strings,
                                &layout.merged_string_start_addresses,
                            )?
                            .wrapping_add(bias)
                            .wrapping_sub(place),
                        RelocationKind::GotRelative => resolution
                            .got_address_for_relocation()?
                            .wrapping_add(addend as u64)
                            .wrapping_add(bias)
                            .wrapping_sub(place),
                        RelocationKind::TlsGd => resolution
                            .tlsgd_got_address()?
                            .wrapping_add(addend as u64)
                            .wrapping_add(bias)
                            .wrapping_sub(place),
                        RelocationKind::TlsLd => layout
                            .prelude()
                            .format_specific
                            .tlsld_got_entry
                            .unwrap()
                            .get()
                            .wrapping_add(addend as u64)
                            .wrapping_add(bias)
                            .wrapping_sub(place),
                        RelocationKind::GotTpOff => resolution
                            .got_address()?
                            .wrapping_add(addend as u64)
                            .wrapping_add(bias)
                            .wrapping_sub(place),
                        _ => bail!(
                            "Unsupported high part relocation {:?} connected with R_RISCV_PCREL_LO12",
                            hi_rel_info.kind
                        ),
                    }
                }
        kind @ (RelocationKind::PairSubtractionULEB128RiscV
        | RelocationKind::PairSubtractionULEB128LoongArch) => {
            let expected_r_type = match kind {
                RelocationKind::PairSubtractionULEB128RiscV => object::elf::R_RISCV_SET_ULEB128,
                RelocationKind::PairSubtractionULEB128LoongArch => object::elf::R_LARCH_ADD_ULEB128,
                _ => unreachable!(),
            };
            get_pair_subtraction_relocation_value::<A, R>(
                object_layout,
                rel,
                layout,
                resolution,
                symbol_index,
                addend,
                // It must be the previous relocation
                &relocation_cache.previous.with_context(|| {
                    "Missing previous relocation for PairSubtractionULEB128".to_owned()
                })?,
                expected_r_type,
            )?
        }
                RelocationKind::GotRelative => resolution
                    .got_address_for_relocation()?
                    .wrapping_add(bias)
                    .wrapping_add(addend as u64)
                    .bitand(mask.got_entry)
                    .wrapping_sub(place.bitand(mask.place)),

                RelocationKind::GotRelativeLoongArch64 => highest_relocation_with_bias(
                    resolution
                        .got_address_for_relocation()?
                        .wrapping_add(addend as u64),
                    place,
                ),
                RelocationKind::GotRelGotBase => resolution
                    .got_address_for_relocation()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                    .wrapping_sub(layout.got_base().bitand(mask.got)),
                RelocationKind::Got => {
                    // The LoongArch64 psABI does not provide a separate GOT Low part relocation for the
                    // TLSGD relocation. So we need to distinguish between a classical GOT
                    // slot and one corresponding to TLSGD.
                    //
                    // Note: TLSLD is unsupported by the target (https://github.com/loongson/la-abi-specs/issues/19).
                    if resolution.flags.needs_got_tls_module() {
                        resolution.tlsgd_got_address()?
                    } else {
                        resolution.got_address_for_relocation()?
                    }
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                }
                RelocationKind::SymRelGotBase => resolution
                    .value_with_addend(
                        addend,
                        symbol_index,
                        object_layout,
                        &layout.symbol_db.section_part_ids,
                        &layout.merged_strings,
                        &layout.merged_string_start_addresses,
                    )?
                    .wrapping_add(bias)
                    .bitand(mask.symbol_plus_addend)
                    .wrapping_sub(layout.got_base().bitand(mask.got)),
                RelocationKind::PltRelGotBase => resolution
                    .plt_address()?
                    .wrapping_add(bias)
                    .wrapping_sub(layout.got_base().bitand(mask.got)),
                RelocationKind::PltRelative => resolution
                    .plt_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .wrapping_sub(place.bitand(mask.place)),
                // TLS-related relocations
                RelocationKind::TlsGd => resolution
                    .tlsgd_got_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                    .wrapping_sub(place.bitand(mask.place)),
                RelocationKind::TlsGdGot => resolution
                    .tlsgd_got_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry),
                RelocationKind::TlsGdGotBase => resolution
                    .tlsgd_got_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                    .wrapping_sub(layout.got_base().bitand(mask.got)),
                RelocationKind::TlsLd => layout
                    .prelude()
                    .format_specific
                    .tlsld_got_entry
                    .unwrap()
                    .get()
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                    .wrapping_sub(place.bitand(mask.place)),
                RelocationKind::TlsLdGot => layout
                    .prelude()
                    .format_specific
                    .tlsld_got_entry
                    .unwrap()
                    .get()
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry),
                RelocationKind::TlsLdGotBase => layout
                    .prelude()
                    .format_specific
                    .tlsld_got_entry
                    .unwrap()
                    .get()
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                    .wrapping_sub(layout.got_base().bitand(mask.got)),
                RelocationKind::DtpOff => resolution
                    .value()
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .wrapping_sub(A::dtp_offset_base(
                        layout,
                        output_kind == OutputKind::SharedObject,
                    )),
                RelocationKind::GotTpOff => resolution
                    .got_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                    .wrapping_sub(place.bitand(mask.place)),
                RelocationKind::GotTpOffLoongArch64 => highest_relocation_with_bias(
                    resolution.got_address()?.wrapping_add(addend as u64),
                    place,
                ),
                RelocationKind::GotTpOffGot => resolution
                    .got_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry),
                RelocationKind::GotTpOffGotBase => resolution
                    .got_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                    .wrapping_sub(layout.got_base().bitand(mask.got)),
                RelocationKind::TpOff
                    if layout
                        .symbol_db
                        .is_undefined(layout.symbol_db.definition(local_symbol_id)) =>
                {
                    // An undefined weak TLS symbol has no offset within the TLS block, so we somewhat
                    // arbitrarily give the 0 offset which at least some other linkers also do and most
                    // importantly is a value guaranteed to fit within the range of any relocation.
                    (addend as u64).wrapping_add(bias)
                }
                RelocationKind::TpOff => resolution
                    .value()
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .wrapping_sub(A::tp_offset_start(layout)),
                RelocationKind::TlsDesc => resolution
                    .tls_descriptor_got_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                    .wrapping_sub(place.bitand(mask.place)),
                RelocationKind::TlsDescLoongArch64 => highest_relocation_with_bias(
                    resolution
                        .tls_descriptor_got_address()?
                        .wrapping_add(addend as u64),
                    place,
                ),
                RelocationKind::TlsDescGot => resolution
                    .tls_descriptor_got_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry),
                RelocationKind::TlsDescGotBase => resolution
                    .tls_descriptor_got_address()?
                    .wrapping_add(addend as u64)
                    .wrapping_add(bias)
                    .bitand(mask.got_entry)
                    .wrapping_sub(layout.got_base().bitand(mask.got)),
        */
        RelocationKind::None | RelocationKind::TlsDescCall => 0,
        // RelocationKind::Alignment
        // | RelocationKind::MachoAddition
        // | RelocationKind::MachoSubtraction => unreachable!(),
        _ => {
            unreachable!()
        }
    };

    let offset_in_section = offset_in_section as usize;

    // Handle addition and subtraction relocation kinds.
    if matches!(
        rel_info.kind,
        RelocationKind::AbsoluteAddition
            | RelocationKind::AbsoluteAdditionWord6
            | RelocationKind::AbsoluteSubtraction
            | RelocationKind::AbsoluteSetWord6
            | RelocationKind::AbsoluteSubtractionWord6
    ) {
        value = rel_info.adjust_value_based_on_content(value, out, offset_in_section)?;
    }

    /*    if let Some(relaxation) = relaxation {
            trace.emit(original_place, || {
                format!(
                    "relaxation applied relaxation={kind:?}, flags={flags},\n\
                    rel_kind={rel_kind:?},\n\
                    value=0x{value:x}, symbol_name={symbol_name}",
                    kind = relaxation.debug_kind(),
                    rel_kind = rel_info.kind,
                    symbol_name = layout.symbol_db.symbol_name_for_display(local_symbol_id),
                )
            });
            tracing::trace!(
                %flags,
                relaxation_kind = ?relaxation.debug_kind(),
                ?rel_info.kind,
                %rel_info.size,
                value,
                value_hex = %HexU64::new(value),
                symbol_name = %layout.symbol_db.symbol_name_for_display(local_symbol_id),
                "relaxation applied");
        } else {
    */
    trace.emit(original_place, || {
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
    // }

    // if let Some(thunked_value) = maybe_get_thunk_for_relocation::<A>(
    //     object_layout,
    //     section_info,
    //     layout,
    //     rel_info,
    //     local_symbol_id,
    //     place,
    //     value,
    // )? {
    //     value = thunked_value;
    // }

    rel_info.write_to_buffer(value, &mut out[offset_in_section..])?;

    // if restore_caller_toc {
    //     A::restore_toc_after_plt_call(out, offset_in_section)?;
    // }

    // Ok(next_modifier)
    Ok(())
}

#[inline(always)]
fn write_absolute_relocation<'data, A: Arch<Platform = Coff>>(
    table_writer: &mut TableWriter<'_, '_>,
    resolution: Resolution<Coff>,
    rel_size: RelocationSize,
    place: u64,
    addend: i64,
    section_info: SectionInfo<<A::Platform as Platform>::SectionFlags>,
    symbol_index: object::SymbolIndex,
    object_layout: &ObjectLayout<'data, Coff>,
    layout: &CoffLayout,
) -> Result<u64> {
    // if !section_info.section_flags.is_alloc() {
    //     resolution.value_with_addend(
    //         addend,
    //         symbol_index,
    //         object_layout,
    //         &layout.symbol_db.section_part_ids,
    //         &layout.merged_strings,
    //         &layout.merged_string_start_addresses,
    //     )
    // } else if resolution.flags.is_dynamic()
    //     && resolution.flags.is_absolute()
    //     && !section_info.is_writable
    // {
    //     // Weak undefined symbol referenced from a read-only section. Fill in as zero.
    //     Ok(0)
    // } else if resolution.flags.is_interposable()
    //     && section_info.is_writable
    //     && rel_size == RelocationSize::ByteSize(C::ADDRESS_SIZE as u8)
    // {
    //     table_writer.write_dynamic_symbol_relocation::<A>(
    //         place,
    //         addend,
    //         resolution.dynamic_symbol_index()?,
    //         DynamicRelocationKind::Absolute,
    //     )?;
    //
    //     Ok(0)
    // } else if resolution.flags.is_ifunc()
    //     && A::absolute_ifunc_needs_irelative(table_writer.output_kind, section_info.is_writable)
    // {
    //     ensure!(
    //         rel_size == RelocationSize::ByteSize(C::ADDRESS_SIZE as u8),
    //         "Relocation against an ifunc is narrower than an address"
    //     );
    //     table_writer
    //         .write_ifunc_relocation_for_data::<A>(place, resolution.raw_value as i64 + addend)?;
    //     Ok(0)
    // } else if table_writer.output_kind.is_position_independent() && !resolution.is_absolute() {
    //     let address = resolution.value_with_addend(
    //         addend,
    //         symbol_index,
    //         object_layout,
    //         &layout.symbol_db.section_part_ids,
    //         &layout.merged_strings,
    //         &layout.merged_string_start_addresses,
    //     )?;
    //     table_writer.write_address_relocation::<A>(place, address)
    // } else {
    //     resolution.value_with_addend(
    //         addend,
    //         symbol_index,
    //         object_layout,
    //         &layout.symbol_db.section_part_ids,
    //         &layout.merged_strings,
    //         &layout.merged_string_start_addresses,
    //     )
    // }
    todo!()
}

fn write_prelude<A: Arch<Platform = coff::Coff>>(
    prelude: &PreludeLayout<coff::Coff>,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    layout: &CoffLayout,
    sym_index_map: &[Option<u32>],
) -> Result {
    verbose_timing_phase!("Write prelude");

    let dos_header: &mut coff::DosHeader =
        from_bytes_mut(buffers.get_mut(crate::part_id::FILE_HEADER))
            .ok()
            .context("Invalid DOS header allocation")?
            .0;
    populate_dos_header::<A>(layout, dos_header)?;

    let dos_stub: &mut [u8; coff::DOS_STUB.len()] =
        from_bytes_mut(buffers.get_mut(part_id::DOS_STUB))
            .ok()
            .context("Invalid DOS header allocation")?
            .0;
    write_dos_stub::<A>(layout, dos_stub);

    let nt_header: &mut coff::NtHeader = from_bytes_mut(buffers.get_mut(coff::part_id::NT_HEADER))
        .ok()
        .context("Invalid NT header allocation")?
        .0;
    populate_nt_header::<A>(layout, &prelude.header_info, nt_header)?;

    let sections_headers: &mut [coff::SectionHeader] = slice_from_bytes_mut(
        buffers.get_mut(coff::part_id::SECTION_HEADERS),
        prelude.header_info.num_output_sections_with_content as usize,
    )
    .ok()
    .context("Invalid section headers allocation")?
    .0;
    populate_section_headers::<A>(layout, &prelude.header_info, sections_headers)?;

    Ok(())
}

fn fill_section_padding<A: Arch<Platform = coff::Coff>>(
    padding: &mut [u8],
    section_info: &SectionOutputInfo<coff::Coff>,
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
