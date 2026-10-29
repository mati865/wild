#![allow(dead_code, unused_variables)]

use crate::OutputFileData;
use crate::coff;
use crate::error::Context;
use crate::file_writer::SizedOutput;
use crate::file_writer::split_buffers_by_alignment;
use crate::file_writer::split_output_by_group;
use crate::file_writer::split_output_into_sections;
use crate::layout::FileLayout;
use crate::layout::Layout;
use crate::output_kind::OutputKind;
use crate::output_section_id::SectionOutputInfo;
use crate::output_section_map::OutputSectionMap;
use crate::output_section_part_map::OutputSectionPartMap;
use crate::output_trace::TraceOutput;
use crate::platform::Arch;
use crate::timing::timing_phase;
use crate::timing::verbose_timing_phase;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::ParallelIterator as _;
use std::ops::Range;
use std::sync::atomic::Ordering::Relaxed;

type CoffLayout<'data> = Layout<'data, coff::Coff>;

pub(crate) fn write<'data, A: Arch<Platform = coff::Coff>>(
    sized_output: &mut SizedOutput<impl OutputFileData>,
    layout: &CoffLayout<'data>,
) -> crate::error::Result {
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
) -> crate::error::Result {
    timing_phase!("Write data to file");
    let (mut section_buffers, padding) = split_output_into_sections(layout, &mut sized_output.out);

    fill_padding_for_sections::<A>(layout, padding);

    let sym_index_map = Vec::new();

    let mut writable_buckets = split_buffers_by_alignment(&mut section_buffers, layout);
    let groups_and_buffers = split_output_by_group(layout, &mut writable_buckets);
    groups_and_buffers
        .into_par_iter()
        .with_max_len(1)
        .try_for_each(|(group, mut buffers)| -> crate::error::Result {
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

fn write_file<'data, A: Arch<Platform = coff::Coff>>(
    file: &FileLayout<'data, coff::Coff>,
    buffers: &mut OutputSectionPartMap<&mut [u8]>,
    table_writer: &mut TableWriter<'_, '_>,
    layout: &CoffLayout<'data>,
    trace: &TraceOutput,
    sym_index_map: &[Option<u32>],
) -> crate::error::Result {
    /*    match file {
            FileLayout::Object(s) => {
                write_object::<A>(s, buffers, table_writer, layout, trace, sym_index_map)?;
            }
            FileLayout::Prelude(s) => {
                write_prelude::<A>(s, buffers, table_writer, layout, sym_index_map)?;
            }
            FileLayout::Epilogue(s) => write_epilogue::<A>(s, buffers, table_writer, layout, trace)?,
            FileLayout::SyntheticSymbols(s) => {
                write_synthetic_symbols::<A>(s, table_writer, layout)?;
            }
            FileLayout::LinkerScript(s) => write_linker_script_state::<A>(s, table_writer, layout)?,
            FileLayout::NotLoaded | FileLayout::StubLibrary(_) => {}
            FileLayout::Dynamic(s) => write_dynamic_file::<A>(s, table_writer, layout)?,
        }
    */
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
        todo!("fill_section_padding uses elf flags")
        // A::fill_section_padding(padding, section_info.section_attributes.flags);
    }
}
