#![allow(dead_code, unused_variables)]

use crate::OutputFileData;
use crate::bail;
use crate::coff;
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
use crate::layout::Section;
use crate::output_kind::OutputKind;
use crate::output_section_id::SectionOutputInfo;
use crate::output_section_map::OutputSectionMap;
use crate::output_section_part_map::OutputSectionPartMap;
use crate::output_trace::TraceOutput;
use crate::platform::Arch;
use crate::platform::Args;
use crate::platform::ObjectFile;
use crate::platform::SectionHeader as _;
use crate::resolution::SectionSlot;
use crate::timing::timing_phase;
use crate::timing::verbose_timing_phase;
use hashbrown::HashMap;
use object::LittleEndian;
use object::from_bytes_mut;
use object::slice_from_bytes_mut;
use object::write::WritableBuffer;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::ParallelIterator as _;
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
    let e = LittleEndian;
    header.e_magic.set(e, object::pe::IMAGE_DOS_SIGNATURE);
    header
        .e_cparhdr
        .set(e, (size_of::<DosHeader>() / u16::BITS as usize).try_into()?);
    header.e_lfanew.set(
        e,
        layout
            .section_layouts
            .get(output_section_id::NT_HEADER)
            .file_offset
            .try_into()?,
    );
    header.e_lfarlc.set(
        e,
        layout
            .section_layouts
            .get(output_section_id::DOS_STUB)
            .file_offset
            .try_into()?,
    );

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
    let mut ty =
        object::pe::IMAGE_FILE_EXECUTABLE_IMAGE | object::pe::IMAGE_FILE_LARGE_ADDRESS_AWARE;
    if output_kind.is_shared_object() {
        ty |= object::pe::IMAGE_FILE_DLL;
    }

    let e = LittleEndian;
    let first_regular_section = layout.section_layouts.get(output_section_id::TEXT);

    header.signature.set(e, object::pe::IMAGE_NT_SIGNATURE);
    header
        .file_header
        .machine
        .set(e, object::pe::IMAGE_FILE_MACHINE_AMD64);
    header
        .file_header
        .number_of_sections
        .set(e, header_info.num_output_sections_with_content.try_into()?);
    header
        .file_header
        .size_of_optional_header
        .set(e, coff::SIZE_OF_OPTIONAL_HEADER64);
    header.file_header.characteristics.set(e, ty);
    header
        .optional_header
        .magic
        .set(e, object::pe::IMAGE_NT_OPTIONAL_HDR64_MAGIC);
    // TODO: Skips a lot of logic compared to ELF one
    header.optional_header.address_of_entry_point = u32::try_from(rva(
        layout.resolved_entry_symbol_address()?.unwrap(),
        layout,
    ))?
    .into();
    header.optional_header.base_of_code.set(
        e,
        u32::try_from(rva(first_regular_section.mem_offset, layout))?,
    );
    header
        .optional_header
        .image_base
        .set(e, layout.args().image_base().unwrap());
    header.optional_header.size_of_image.set(
        e,
        u32::try_from(first_regular_section.alignment.align_up(rva(
            layout.section_layouts.iter().last().unwrap().1.mem_end(),
            layout,
        )))?,
    );
    header.optional_header.size_of_headers.set(
        e,
        u32::try_from(
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
        )?,
    );
    header
        .optional_header
        .section_alignment
        .set(e, first_regular_section.alignment.value().try_into()?);
    // Wild doesn't support different memory and file alignment
    header
        .optional_header
        .file_alignment
        .set(e, first_regular_section.alignment.value().try_into()?);
    // TODO: filter only sections with CODE characteristics
    // header.optional_header.size_of_initialized_data.set(
    //     e,
    //     u32::try_from(
    //         layout
    //             .section_layouts
    //             .iter()
    //             .map(|(_, sec)| sec.file_size)
    //             .sum::<usize>(),
    //     )?,
    // );
    header.optional_header.size_of_initialized_data.set(
        e,
        u32::try_from(
            layout
                .section_layouts
                .get(output_section_id::TEXT)
                .file_size,
        )?,
    );
    header.optional_header.size_of_uninitialized_data.set(
        e,
        u32::try_from(layout.section_layouts.get(output_section_id::BSS).mem_size)?,
    );

    // TODO: hardcoded based LLD for sane defaults for now
    header.optional_header.major_linker_version = 14;
    header.optional_header.minor_linker_version = 0;
    header
        .optional_header
        .major_operating_system_version
        .set(e, 6);
    header
        .optional_header
        .minor_operating_system_version
        .set(e, 0);
    header.optional_header.major_image_version.set(e, 0);
    header.optional_header.minor_image_version.set(e, 0);
    header.optional_header.major_subsystem_version.set(e, 6);
    header.optional_header.minor_subsystem_version.set(e, 0);
    header
        .optional_header
        .subsystem
        .set(e, object::pe::IMAGE_SUBSYSTEM_WINDOWS_CUI);
    header
        .optional_header
        .number_of_rva_and_sizes
        .set(e, object::pe::IMAGE_NUMBEROF_DIRECTORY_ENTRIES.try_into()?);
    // TODO: other fields

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

    // let relocations = object.relocations(section_index)?;
    // let result = match relocations {
    //     elf::RelocationList::Rela(rela) => apply_relocations::<C, A, elf::ElfRela<C>, _>(
    //         object,
    //         out,
    //         section_index,
    //         rela.iter().map(|rela| Ok(elf::ElfRela::new(*rela))),
    //         &instruction_mapping,
    //         layout,
    //         table_writer,
    //         trace,
    //     ),
    //     elf::RelocationList::Crel(crel_iter) => apply_relocations::<C, A, elf::ElfCrel<C>, _>(
    //         object,
    //         out,
    //         section_index,
    //         crel_iter.map(|r| r.map(elf::ElfCrel::new)),
    //         &instruction_mapping,
    //         layout,
    //         table_writer,
    //         trace,
    //     ),
    // };
    // result.with_context(|| {
    //     format!(
    //         "Failed to apply relocations in section `{}` of {}",
    //         object.object.section_display_name(section_index),
    //         object.input
    //     )
    // })?;
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
