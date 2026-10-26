#![allow(dead_code, unused_variables)]

use crate::FileSystem;
use crate::arch::Architecture;
use crate::args::coff::CoffArgs;
use crate::bail;
use crate::error::Result;
use crate::input_data::InputBytes;
use crate::input_data::InputRef;
use crate::layout;
use crate::layout_rules;
use crate::output_section_id::OutputSectionId;
use crate::output_section_id::SectionIdentity;
use crate::output_section_id::SectionName;
use crate::output_section_part_map::OutputSectionPartMap;
use crate::platform;
use crate::platform::Arch;
use crate::platform::Args;
use crate::platform::CommonSymbol;
use crate::platform::Platform;
use crate::resolution::LoadedMetrics;
use crate::symbol_db::SymbolDb;
use crate::symbol_db::Visibility;
use object::LittleEndian;
use object::SymbolIndex;
use object::coff::CoffHeader;
use std::borrow::Cow;
use std::fmt::Display;
use std::fmt::Formatter;
use std::marker::PhantomData;

pub(crate) fn link_for_arch<'data, F: FileSystem>(
    linker: &'data crate::Linker<F>,
    args: &'data CoffArgs,
) -> Result<crate::LinkerOutput<'data>> {
    match args.architecture() {
        Architecture::X86_64 => linker.link_for_arch::<Coff, crate::coff_x86_64::CoffX86_64>(args),
        _ => {
            todo!()
        }
    }
}

pub(crate) type SymtabEntry = object::pe::ImageSymbol;
pub(crate) type SectionHeader = object::pe::ImageSectionHeader;
pub(crate) type DynamicEntry = ();
pub(crate) type RelocationSections = ();
pub(crate) type SymbolTable<'data> = object::read::coff::SymbolTable<'data>;
pub(crate) type VersionNames = ();

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct Coff;

#[repr(u32)]
#[derive(Clone, Copy)]
enum SinglePartSectionId {
    ProgramHeaders = crate::output_section_id::NUM_COMMON_SINGLE_PART_SECTIONS,
    SectionHeaders,
    Shstrtab,
    Strtab,
    Got,
    GotRelr,
    PltGot,
    RelaPlt,
    EhFrame,
    EhFrameHdr,
    Sframe,
    Dynamic,
    SysvHash,
    GnuHash,
    Dynsym,
    Dynstr,
    Interp,
    GnuVersion,
    GnuVersionD,
    GnuVersionR,
    NoteGnuProperty,
    NoteGnuBuildId,
    SymtabLocal,
    SymtabGlobal,
    RelaDynRelative,
    RelaDynGeneral,
    RiscvAttributes,
    RelroPadding,
    RelrDyn,
    SymtabShndxLocal,
    SymtabShndxGlobal,
    GdbIndex,

    // Must be last.
    Count,
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum RegularSectionId {
    Rodata,
    InitArray,
    FiniArray,
    PreinitArray,
    Text,
    Init,
    Fini,
    Data,
    Tdata,
    Tbss,
    Bss,
    Comment,
    GccExceptTable,
    NoteAbiTag,
    DataRelRo,
    PartialLinkingSingletons,

    // Must be last.
    Count,
}

#[derive(derive_more::Debug)]
pub(crate) struct File<'data> {
    pub(crate) arch: Architecture,
    #[debug(skip)]
    pub(crate) data: &'data [u8],
    pub(crate) sections: object::coff::SectionTable<'data>,
    pub(crate) symbols: object::read::coff::SymbolTable<'data>,
}

impl<'data> File<'data> {
    fn parse_coff_bytes(data: &'data [u8], is_dynamic: bool) -> Result<Self> {
        let mut offset = 0;
        let header = object::pe::ImageFileHeader::parse(data, &mut offset)?;
        let arch = match header.machine() {
            object::pe::IMAGE_FILE_MACHINE_AMD64 => Architecture::X86_64,
            _ => bail!("Unsupported architecture: {:?}", header.machine()),
        };

        let sections = header.sections(data, offset)?;
        let symbols = header.symbols(data)?;

        Ok(Self {
            arch,
            data,
            sections,
            symbols,
        })
    }
}

const ELF_NUM_SINGLE_PART_SECTIONS: u32 = SinglePartSectionId::Count as u32;
const ELF_NUM_BUILT_IN_REGULAR_SECTIONS: usize = RegularSectionId::Count as usize;
const ELF_NUM_BUILT_IN_SECTIONS: usize =
    ELF_NUM_SINGLE_PART_SECTIONS as usize + ELF_NUM_BUILT_IN_REGULAR_SECTIONS;

impl Platform for Coff {
    const NUM_SINGLE_PART_SECTIONS: u32 = SinglePartSectionId::Count as u32;
    const NUM_BUILT_IN_REGULAR_SECTIONS: usize = 0;

    const VERIFY_IGNORE_SECTION_IDS: &'static [OutputSectionId] =
        &[crate::output_section_id::FILE_HEADER];

    type File<'data> = File<'data>;
    type FileFlags = u32;
    type SymtabEntry = SymtabEntry;
    type PlatformSpecificSymbol = ();
    type SectionHeader = SectionHeader;
    type SectionFlags = SectionFlags;
    type SectionAttributes = SectionAttributes;
    type SectionType = SectionType;
    type SegmentType = SegmentType;
    type ProgramSegmentDef = ProgramSegmentDef;
    type BuiltInSectionDetails = BuiltInSectionDetails;
    type RelocationSections = ();
    type DynamicEntry = ();
    type DynamicSymbolDefinitionExt = ();
    type RelocationInfo = RelocationType;
    type NonAddressableIndexes = NonAddressableIndexes;
    type NonAddressableCounts = ();
    type EpilogueLayoutExt = ();
    type GroupLayoutExt = ();
    type CommonGroupStateExt = ();
    type StubLibraryLayoutStateExt = ();
    type StubLibraryLayoutExt = ();
    type ArchIdentifier = ();
    type Args = CoffArgs;
    type ResolutionExt = ();
    type SymtabShndxEntry = ();
    type SymbolVersionIndex = ();
    type FinaliseSizesExt<'data> = ();
    type LayoutExt<'data> = ();
    type GdbIndexScanResult<'data> = ();
    type SectionIterator<'a> = core::slice::Iter<'a, SectionHeader>;
    type DynamicTagValues<'data> = DynamicTagValues<'data>;
    type RelocationList<'data> = RelocationList<'data>;
    type DynamicLayoutStateExt<'data> = ();
    type DynamicLayoutExt<'data> = ();
    type LayoutResourcesExt<'data> = ();
    type PreludeLayoutStateExt = ();
    type PreludeLayoutExt = ();
    type ObjectLayoutStateExt<'data> = ();
    type RawSymbolName<'data> = RawSymbolName<'data>;
    type VersionNames<'data> = ();
    type VerneedTable<'data> = VerneedTable<'data>;
    type ResolvedObjectExt<'data> = ();
    type SectionIdentityExt = ();
    type GcUnit = ();

    fn write_output_file<'data, A: Arch<Platform = Self>, F: FileSystem>(
        output: &crate::file_writer::Output<F>,
        layout: &layout::Layout<'data, Self>,
    ) -> Result {
        todo!()
    }

    fn section_attributes(_header: &Self::SectionHeader) -> Self::SectionAttributes {
        SectionAttributes::default()
    }

    fn apply_force_keep_sections(
        _keep_sections: &mut crate::output_section_map::OutputSectionMap<bool>,
        _args: &Self::Args,
    ) {
        // TODO
    }

    fn is_zero_sized_section_content(_section_id: OutputSectionId) -> bool {
        // TODO
        false
    }

    fn built_in_section_details() -> &'static [Self::BuiltInSectionDetails] {
        // TODO
        // &crate::coff::SECTION_DEFINITIONS
        &[]
    }

    fn finalise_group_layout(_memory_offsets: &OutputSectionPartMap<u64>) -> Self::GroupLayoutExt {}

    fn frame_data_base_address(_memory_offsets: &OutputSectionPartMap<u64>) -> u64 {
        0
    }

    fn post_gc<'data>(
        groups: &mut [layout::GroupState<Self>],
        _symbol_db: &SymbolDb<'data, Self>,
    ) -> Result {
        // TODO
        Ok(())
    }

    fn activate_dynamic<'data>(
        _state: &mut layout::DynamicLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
    ) {
    }

    fn pre_finalise_sizes_prelude<'scope, 'data>(
        _prelude: &mut layout::PreludeLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
        _resources: &layout::GraphResources<'data, 'scope, Self>,
    ) {
    }

    fn finalise_sizes_dynamic<'data>(
        _object: &mut layout::DynamicLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
    ) -> Result {
        Ok(())
    }

    fn finalise_object_sizes<'data>(
        _object: &mut layout::ObjectLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
    ) {
    }

    fn finalise_object_layout<'data>(
        _object: &layout::ObjectLayoutState<'data, Self>,
        _memory_offsets: &mut OutputSectionPartMap<u64>,
    ) {
    }

    fn finalise_layout_dynamic<'data>(
        _state: &mut layout::DynamicLayoutState<'data, Self>,
        _memory_offsets: &mut OutputSectionPartMap<u64>,
        _resources: &layout::FinaliseLayoutResources<'_, 'data, Self>,
        _resolutions_out: &mut layout::ResolutionWriter<Self>,
    ) -> Result<Option<Self::DynamicLayoutExt<'data>>> {
        Ok(None)
    }

    fn take_dynsym_index(
        _memory_offsets: &mut OutputSectionPartMap<u64>,
        _section_layouts: &crate::output_section_map::OutputSectionMap<layout::OutputRecordLayout>,
    ) -> Result<u32> {
        todo!()
    }

    fn compute_object_addresses<'data>(
        _object: &layout::ObjectLayoutState<'data, Self>,
        _memory_offsets: &mut OutputSectionPartMap<u64>,
    ) {
    }

    fn layout_resources_ext<'data>(
        _groups: &[crate::grouping::Group<'data, Self>],
    ) -> Self::LayoutResourcesExt<'data> {
    }

    fn gc_unit_for_symbol<'data>(
        object: &Self::File<'data>,
        symbol: &Self::SymtabEntry,
        _symbol_index: SymbolIndex,
    ) -> Result<Option<Self::GcUnit>> {
        // TODO
        Ok(None)
    }

    fn activate_object_gc<'data, 'scope, A: Arch<Platform = Self>>(
        object: &mut layout::ObjectLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
        resources: &'scope layout::GraphResources<'data, 'scope, Self>,
        queue: &mut layout::LocalWorkQueue<Self>,
        scope: &rayon::Scope<'scope>,
    ) -> Result {
        // TODO
        Ok(())
    }

    fn load_gc_unit<'data, 'scope, A: Arch<Platform = Self>>(
        object: &mut layout::ObjectLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
        resources: &'scope layout::GraphResources<'data, 'scope, Self>,
        queue: &mut layout::LocalWorkQueue<Self>,
        unit: Self::GcUnit,
        scope: &rayon::Scope<'scope>,
    ) -> Result {
        // TODO
        Ok(())
    }

    fn load_object_section_relocations<'data, 'scope, A: Arch<Platform = Self>>(
        _state: &mut layout::ObjectLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
        _queue: &mut layout::LocalWorkQueue<Self>,
        _resources: &'scope layout::GraphResources<'data, '_, Self>,
        _section: layout::Section,
        _section_index: object::SectionIndex,
        _scope: &rayon::Scope<'scope>,
    ) -> Result {
        Ok(())
    }

    fn create_dynamic_symbol_definition<'data>(
        _symbol_db: &SymbolDb<'data, Self>,
        _symbol_id: crate::symbol_db::SymbolId,
    ) -> Result<layout::DynamicSymbolDefinition<'data, Self>> {
        todo!()
    }

    fn update_segment_keep_list(
        _program_segments: &crate::program_segments::ProgramSegments<Self::ProgramSegmentDef>,
        _keep_segments: &mut [bool],
        _args: &Self::Args,
    ) {
    }

    fn program_segment_defs() -> &'static [Self::ProgramSegmentDef] {
        // TODO
        &[]
    }

    fn unconditional_segment_defs() -> &'static [Self::ProgramSegmentDef] {
        &[]
    }

    fn program_segment_should_include_section(
        segment_def: Self::ProgramSegmentDef,
        _section_info: &crate::output_section_id::SectionOutputInfo<Self>,
        section_id: OutputSectionId,
        _rosegment: bool,
    ) -> bool {
        // TODO
        true
    }

    fn create_linker_defined_symbols(
        symbols: &mut crate::parsing::InternalSymbolsBuilder<Self>,
        _output_kind: crate::output_kind::OutputKind,
        args: &Self::Args,
    ) {
        // TODO
    }

    fn built_in_section_infos<'data>()
    -> Vec<crate::output_section_id::SectionOutputInfo<'data, Self>> {
        // TODO
        Vec::new()
    }

    fn new_resolved_object_ext<'data>(
        symbol_id_range: crate::symbol_db::SymbolIdRange,
        file_id: crate::input_data::FileId,
    ) -> Self::ResolvedObjectExt<'data> {
        todo!()
    }

    fn new_object_layout_state_ext<'data>(
        input: Self::ResolvedObjectExt<'data>,
    ) -> Self::ObjectLayoutStateExt<'data> {
        input
    }

    fn create_finalise_sizes_ext<'data, 'states, 'files, A: Arch<Platform = Self>>(
        _args: &Self::Args,
        groups: &'files mut [layout::GroupState<'data, Self>],
        symbol_db: &SymbolDb<'data, Self>,
    ) -> Result<Self::FinaliseSizesExt<'data>>
    where
        'data: 'files,
        'data: 'states,
    {
        todo!()
    }

    fn create_layout_ext<'data>(
        finalise_sizes_ext: Self::FinaliseSizesExt<'data>,
        _resolutions: &layout::SymbolResolutions<Self>,
        _group_layouts: &[layout::GroupLayout<'data, Self>],
    ) -> Result<Self::LayoutExt<'data>> {
        Ok(finalise_sizes_ext)
    }

    fn load_exception_frame_data<'data, 'scope, A: Arch<Platform = Self>>(
        _object: &mut layout::ObjectLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
        _eh_frame_section_index: object::SectionIndex,
        _resources: &'scope layout::GraphResources<'data, '_, Self>,
        _queue: &mut layout::LocalWorkQueue<Self>,
        _scope: &rayon::Scope<'scope>,
    ) -> Result {
        // PE doesn't have ELF-style `.eh_frame`.
        Ok(())
    }

    fn non_empty_section_loaded<'data, 'scope, A: Arch<Platform = Self>>(
        _object: &mut layout::ObjectLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
        _queue: &mut layout::LocalWorkQueue<Self>,
        _unloaded: crate::resolution::UnloadedSection,
        _resources: &'scope layout::GraphResources<'data, 'scope, Self>,
        _scope: &rayon::Scope<'scope>,
    ) -> Result {
        Ok(())
    }

    fn new_epilogue_layout<'data>(
        _args: &Self::Args,
        _output_kind: crate::output_kind::OutputKind,
        _dynamic_symbol_definitions: &mut [layout::DynamicSymbolDefinition<'data, Self>],
        _group_states: &[layout::GroupState<'data, Self>],
    ) -> Self::EpilogueLayoutExt {
    }

    fn apply_non_addressable_indexes_epilogue(
        _counts: &mut Self::NonAddressableCounts,
        _state: &mut Self::EpilogueLayoutExt,
    ) {
        // No-op: PE has no version table.
    }

    fn apply_non_addressable_indexes<'data, 'groups>(
        _symbol_db: &SymbolDb<'data, Self>,
        _counts: &Self::NonAddressableCounts,
        _mem_sizes_iter: impl Iterator<Item = &'groups mut OutputSectionPartMap<u64>>,
    ) {
        todo!()
    }

    fn finalise_sizes_epilogue<'data>(
        _state: &mut Self::EpilogueLayoutExt,
        mem_sizes: &mut OutputSectionPartMap<u64>,
        _dynamic_symbol_definitions: &[layout::DynamicSymbolDefinition<'data, Self>],
        properties: &Self::LayoutExt<'data>,
        _symbol_db: &SymbolDb<'data, Self>,
    ) -> Result<()> {
        // TODO
        Ok(())
    }

    fn finalise_sizes_all<'data>(
        _mem_sizes: &mut OutputSectionPartMap<u64>,
        _symbol_db: &SymbolDb<'data, Self>,
    ) {
    }

    fn finalise_layout_epilogue<'data>(
        _epilogue_state: &mut Self::EpilogueLayoutExt,
        memory_offsets: &mut OutputSectionPartMap<u64>,
        _symbol_db: &SymbolDb<'data, Self>,
        common_state: &Self::LayoutExt<'data>,
        _dynsym_start_index: u32,
        _dynamic_symbol_defs: &[layout::DynamicSymbolDefinition<Self>],
    ) -> Result {
        // TODO
        Ok(())
    }

    fn is_symbol_non_interposable<'data>(
        _object: &Self::File<'data>,
        _args: &Self::Args,
        _sym: &Self::SymtabEntry,
        _output_kind: crate::output_kind::OutputKind,
        _export_list: Option<&crate::export_list::ExportList>,
        _lib_name: &[u8],
        _archive_semantics: bool,
        _is_undefined: bool,
    ) -> bool {
        // TODO
        false
    }

    fn allocate_header_sizes<'data>(
        _prelude: &mut layout::PreludeLayoutState<'data, Self>,
        sizes: &mut OutputSectionPartMap<u64>,
        _header_info: &layout::HeaderInfo,
        _program_segments: &crate::program_segments::ProgramSegments<Self::ProgramSegmentDef>,
        _output_sections: &crate::output_section_id::OutputSections<Self>,
        _resources: &layout::FinaliseSizesResources<'data, '_, Self>,
        _args: &Self::Args,
    ) {
        sizes.increment(
            crate::part_id::FILE_HEADER,
            object::pe::IMAGE_SIZEOF_FILE_HEADER as u64,
        );

        // TODO
        //     sizes.increment(
        //         part_id::PROGRAM_HEADERS,
        //         program_headers_size::<C>(header_info),
        //     );
        //
        //     sizes.increment(
        //         part_id::SECTION_HEADERS,
        //         section_headers_size::<C>(header_info),
        //     );
        //
        //     let regular_strtab_size = output_sections
        //         .ids_with_info()
        //         .filter(|(id, _info)| output_sections.output_index_of_section(*id).is_some())
        //         .map(|(_id, info)| {
        //             if let SectionKind::Primary(identity) = info.kind {
        //                 identity.section_name().len() as u64 + 1
        //             } else {
        //                 0
        //             }
        //         })
        //         .sum::<u64>();
        //
        //     prelude.format_specific.shstrtab_size =
        //         regular_strtab_size + header_info.partial_link_section_name_bytes;
        //
        //     sizes.increment(part_id::SHSTRTAB, prelude.format_specific.shstrtab_size);
    }

    fn finalise_sizes_for_symbol<'data>(
        _common: &mut layout::CommonGroupState<'data, Self>,
        _symbol_db: &SymbolDb<'data, Self>,
        _symbol_id: crate::symbol_db::SymbolId,
        _flags: crate::value_flags::ValueFlags,
    ) -> Result {
        Ok(())
    }

    fn allocate_resolution(
        _flags: crate::value_flags::ValueFlags,
        _mem_sizes: &mut OutputSectionPartMap<u64>,
        _output_kind: crate::output_kind::OutputKind,
        _args: &Self::Args,
    ) {
    }

    fn allocate_object_symtab_space<'data>(
        _state: &layout::ObjectLayoutState<'data, Self>,
        _common: &mut layout::CommonGroupState<'data, Self>,
        _symbol_db: &SymbolDb<'data, Self>,
        _per_symbol_flags: &crate::value_flags::AtomicPerSymbolFlags,
    ) -> Result {
        Ok(())
    }

    fn allocate_internal_symbol(
        _symbol_id: crate::symbol_db::SymbolId,
        _def_info: &crate::parsing::InternalSymDefInfo<Self>,
        _sizes: &mut OutputSectionPartMap<u64>,
        _symbol_db: &SymbolDb<Self>,
    ) -> Result {
        Ok(())
    }

    fn allocate_prelude(_common: &mut layout::CommonGroupState<Self>, _symbol_db: &SymbolDb<Self>) {
    }

    fn finalise_prelude_layout<'data>(
        _prelude: &layout::PreludeLayoutState<Self>,
        _memory_offsets: &mut OutputSectionPartMap<u64>,
        _resources: &layout::FinaliseLayoutResources<'_, 'data, Self>,
    ) -> Result<Self::PreludeLayoutExt> {
        Ok(())
    }

    fn create_resolution(
        flags: crate::value_flags::ValueFlags,
        raw_value: u64,
        dynamic_symbol_index: Option<std::num::NonZeroU32>,
        _memory_offsets: &mut OutputSectionPartMap<u64>,
        _args: &<Self as Platform>::Args,
        _output_kind: crate::OutputKind,
    ) -> layout::Resolution<Self> {
        layout::Resolution {
            raw_value,
            dynamic_symbol_index,
            flags,
            format_specific: (),
        }
    }

    fn raw_symbol_name<'data>(
        name_bytes: &'data [u8],
        _verneed_table: &Self::VerneedTable<'data>,
        _symbol_index: SymbolIndex,
    ) -> Self::RawSymbolName<'data> {
        RawSymbolName { name: name_bytes }
    }

    fn default_layout_rules(_args: &Self::Args) -> Vec<layout_rules::SectionRule<'static>> {
        // TODO
        Vec::new()
    }

    fn align_load_segment_start(
        _segment_def: Self::ProgramSegmentDef,
        _segment_alignment: crate::alignment::Alignment,
        _file_offset: &mut usize,
        _mem_offset: &mut u64,
    ) {
    }

    fn build_output_order_and_program_segments<'data>(
        _custom: &crate::output_section_id::CustomSectionIds,
        output_kind: crate::output_kind::OutputKind,
        output_sections: &crate::output_section_id::OutputSections<'data, Self>,
        secondary: &crate::output_section_map::OutputSectionMap<Vec<OutputSectionId>>,
        _location_counters: &[layout_rules::LocationCounter<'data>],
    ) -> (
        crate::output_section_id::OutputOrder<'data>,
        crate::program_segments::ProgramSegments<Self::ProgramSegmentDef>,
    ) {
        todo!()
    }

    fn default_symtab_entry() -> Self::SymtabEntry {
        todo!()
    }

    fn is_allowed_in_archive(kind: crate::file_kind::FileKind) -> bool {
        kind == crate::file_kind::FileKind::Amd64PeObject
    }

    fn section_identity<'data>(
        name: SectionName<'data>,
        _section: &Self::SectionHeader,
    ) -> SectionIdentity<'data, Self> {
        SectionIdentity::new(name, ())
    }

    fn section_identity_from_name<'data>(
        name: SectionName<'data>,
    ) -> Option<SectionIdentity<'data, Self>> {
        Some(SectionIdentity::new(name, ()))
    }
}

impl<'data> platform::ObjectFile<'data> for File<'data> {
    type Platform = Coff;

    fn section_groups(&self) -> Result<Vec<crate::comdat::SectionGroup<'data>>> {
        // TODO
        Ok(vec![])
    }

    fn parse(input: &InputBytes<'data>, args: &CoffArgs) -> Result<Self> {
        let file = Self::parse_bytes(input.data, false)?;

        if file.arch != args.architecture() {
            bail!(
                "`{}` has incompatible architecture: {}, expecting {}",
                input,
                file.arch,
                args.architecture(),
            )
        }

        Ok(file)
    }

    fn parse_bytes(data: &'data [u8], is_dynamic: bool) -> Result<Self> {
        Self::parse_coff_bytes(data, is_dynamic)
    }

    fn section(&self, index: object::SectionIndex) -> Result<&SectionHeader> {
        todo!()
    }

    fn section_by_name(&self, name: &str) -> Option<(object::SectionIndex, &SectionHeader)> {
        todo!()
    }

    fn section_name(&self, index: object::SectionIndex) -> Result<&'data [u8]> {
        todo!()
    }

    fn section_display_name(&self, index: object::SectionIndex) -> Cow<'data, str> {
        self.section_name(index).map_or_else(
            |_| format!("<index {}>", index.0).into(),
            String::from_utf8_lossy,
        )
    }

    fn raw_section_data(&self, section: &SectionHeader) -> Result<&'data [u8]> {
        todo!()
    }

    fn section_data(
        &self,
        section: &SectionHeader,
        member: &bumpalo_herd::Member<'data>,
        loaded_metrics: &LoadedMetrics,
    ) -> Result<&'data [u8]> {
        todo!()
    }

    fn copy_section_data(&self, section: &SectionHeader, out: &mut [u8]) -> Result {
        todo!()
    }

    fn section_data_cow(&self, section: &SectionHeader) -> Result<Cow<'data, [u8]>> {
        todo!()
    }

    fn section_size(&self, section: &SectionHeader) -> Result<u64> {
        todo!()
    }

    fn section_alignment(&self, section: &SectionHeader) -> Result<u64> {
        todo!()
    }

    fn relocations(
        &self,
        index: object::SectionIndex,
        relocations: &RelocationSections,
    ) -> Result<RelocationList<'data>> {
        todo!()
    }

    fn symbol(&self, index: SymbolIndex) -> Result<&SymtabEntry> {
        Ok(self.symbols.symbol(index)?)
    }

    fn symbol_name(&self, symbol: &SymtabEntry) -> Result<&'data [u8]> {
        todo!()
    }

    fn symbol_section(
        &self,
        symbol: &SymtabEntry,
        index: SymbolIndex,
    ) -> Result<Option<object::SectionIndex>> {
        todo!()
    }

    fn dynamic_tags(&self) -> Result<&'data [DynamicEntry]> {
        Ok(&[])
    }

    fn parse_relocations(&self) -> Result<RelocationSections> {
        todo!()
    }

    fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    fn is_dynamic(&self) -> bool {
        // TODO
        false
    }

    fn dynamic_tag_values(&self) -> Option<DynamicTagValues<'data>> {
        None
    }

    fn symbol_version_debug(&self, symbol_index: SymbolIndex) -> Option<String> {
        None
    }

    fn section_iter<'a>(&'a self) -> core::slice::Iter<'a, SectionHeader> {
        self.sections.iter()
    }

    fn enumerate_sections(&self) -> impl Iterator<Item = (object::SectionIndex, &SectionHeader)> {
        self.sections.enumerate()
    }

    fn get_version_names(&self) -> Result<VersionNames> {
        Ok(())
    }

    fn get_symbol_name_and_version(
        &self,
        symbol: &SymtabEntry,
        local_index: usize,
        version_names: &VersionNames,
    ) -> Result<RawSymbolName<'data>> {
        let name_bytes = self.symbol_name(symbol)?;

        Ok(RawSymbolName { name: name_bytes })
    }

    fn symbols_iter(&self) -> impl Iterator<Item = &SymtabEntry> {
        self.symbols.iter().map(|(_, sym)| sym)
    }

    fn verneed_table(&self) -> Result<VerneedTable<'data>> {
        Ok(VerneedTable::default())
    }

    fn num_sections(&self) -> usize {
        self.sections.len()
    }

    fn process_gnu_note_section(
        &self,
        state: &mut (),
        section_index: object::SectionIndex,
    ) -> Result {
        todo!()
    }

    fn symbol_versions(&self) -> &[()] {
        &[]
    }

    fn dynamic_symbol_used(
        &self,
        symbol_index: SymbolIndex,
        file: &mut layout::DynamicLayoutState<'data, Coff>,
    ) -> Result {
        todo!()
    }

    fn finalise_sizes_dynamic(
        &self,
        lib_name: &[u8],
        state: &mut (),
        mem_sizes: &mut OutputSectionPartMap<u64>,
    ) -> Result {
        todo!()
    }

    fn apply_non_addressable_indexes_dynamic(
        &self,
        indexes: &mut NonAddressableIndexes,
        counts: &mut (),
        state: &mut (),
    ) -> Result {
        todo!()
    }

    fn should_enforce_undefined(
        &self,
        _resources: &layout::GraphResources<'data, '_, Coff>,
    ) -> bool {
        false
    }

    fn symbol_offset_in_section(
        &self,
        symbol: &SymtabEntry,
        _section_index: object::SectionIndex,
    ) -> Result<u64> {
        todo!()
    }
}

impl platform::Symbol for SymtabEntry {
    fn as_common(&self) -> Option<CommonSymbol> {
        None
    }

    fn is_undefined(&self) -> bool {
        self.section_number.get(LittleEndian) == object::pe::IMAGE_SYM_UNDEFINED.0 as u16
    }

    fn is_local(&self) -> bool {
        false
    }

    fn visibility(&self) -> Visibility {
        Visibility::Default
    }

    fn is_absolute(&self) -> bool {
        self.section_number.get(LittleEndian) == object::pe::IMAGE_SYM_ABSOLUTE.0 as u16
    }

    fn is_weak(&self) -> bool {
        // TODO
        false
    }

    fn value(&self) -> u64 {
        self.value.get(LittleEndian) as u64
    }

    fn size(&self) -> u64 {
        // TODO
        0
    }

    fn has_name(&self) -> bool {
        // TODO
        true
    }

    fn is_default_strippable(&self, name: &[u8]) -> bool {
        // TODO
        false
    }

    fn debug_string(&self) -> String {
        // TODO
        // crate::elf::SymDebug(self).to_string()
        String::new()
    }

    fn is_tls(&self) -> bool {
        // TODO
        false
    }

    fn is_interposable(&self) -> bool {
        // TODO
        false
    }

    fn is_func(&self) -> bool {
        // TODO
        false
    }

    fn is_ifunc(&self) -> bool {
        // TODO
        false
    }

    fn is_hidden(&self) -> bool {
        // TODO
        false
    }

    fn is_gnu_unique(&self) -> bool {
        // TODO
        false
    }

    fn with_hidden(self, _hidden: bool) -> Self {
        // TODO: ?
        self
    }
}

// TODO
impl platform::SectionHeader for SectionHeader {
    fn is_alloc(&self) -> bool {
        false
    }

    fn is_writable(&self) -> bool {
        false
    }

    fn is_executable(&self) -> bool {
        false
    }

    fn is_tls(&self) -> bool {
        false
    }

    fn is_merge_section(&self) -> bool {
        false
    }

    fn is_strings(&self) -> bool {
        false
    }

    fn should_retain(&self) -> bool {
        false
    }

    fn should_exclude(&self) -> bool {
        false
    }

    fn is_group(&self) -> bool {
        false
    }

    fn is_note(&self) -> bool {
        false
    }

    fn is_prog_bits(&self) -> bool {
        false
    }

    fn is_no_bits(&self) -> bool {
        false
    }

    fn is_null(&self) -> bool {
        false
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct SectionFlags {}

impl platform::SectionFlags for SectionFlags {
    fn is_alloc(self) -> bool {
        // TODO
        false
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct SectionType {}

// TODO
impl platform::SectionType for SectionType {
    fn is_rela(&self) -> bool {
        false
    }

    fn is_rel(&self) -> bool {
        false
    }

    fn is_symtab(&self) -> bool {
        false
    }

    fn is_strtab(&self) -> bool {
        false
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct SectionAttributes {}

impl platform::SectionAttributes for SectionAttributes {
    type Platform = Coff;

    fn merge(&mut self, _rhs: Self) {
        // TODO
    }

    fn apply(
        &self,
        _output_sections: &mut crate::output_section_id::OutputSections<Self::Platform>,
        _section_id: OutputSectionId,
    ) {
        // TODO
    }

    fn is_null(&self) -> bool {
        false
    }

    fn is_alloc(&self) -> bool {
        false
    }

    fn is_executable(&self) -> bool {
        false
    }

    fn is_tls(&self) -> bool {
        false
    }

    fn occupies_only_tls_address_space(&self) -> bool {
        false
    }

    fn is_writable(&self) -> bool {
        false
    }

    fn is_no_bits(&self) -> bool {
        false
    }

    fn flags(&self) -> <Self::Platform as Platform>::SectionFlags {
        SectionFlags::default()
    }

    fn ty(&self) -> <Self::Platform as Platform>::SectionType {
        SectionType::default()
    }

    fn set_to_default_type(&mut self) {}
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct SegmentType {}

impl platform::SegmentType for SegmentType {}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct ProgramSegmentDef {}

// TODO
impl platform::ProgramSegmentDef for ProgramSegmentDef {
    fn is_writable(self) -> bool {
        false
    }

    fn is_executable(self) -> bool {
        false
    }

    fn always_keep(self) -> bool {
        false
    }

    fn is_loadable(self) -> bool {
        false
    }

    fn is_stack(self) -> bool {
        false
    }

    fn is_tls(self) -> bool {
        false
    }

    fn order_key(self) -> usize {
        0
    }
}

impl Display for ProgramSegmentDef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct BuiltInSectionDetails {}

// TODO
impl platform::BuiltInSectionDetails for BuiltInSectionDetails {}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct RelocationType {}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct NonAddressableIndexes {}

impl platform::NonAddressableIndexes for NonAddressableIndexes {
    fn new<P: Platform>(symbol_db: &SymbolDb<P>) -> Self {
        Self::default()
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct DynamicTagValues<'data> {
    _phantom: PhantomData<&'data ()>,
}

impl<'data> platform::DynamicTagValues<'data> for DynamicTagValues<'data> {
    fn lib_name(&self, input: &InputRef<'data>) -> &'data [u8] {
        todo!()
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct RelocationList<'data> {
    _phantom: PhantomData<&'data ()>,
}

impl<'data> platform::RelocationList<'data> for RelocationList<'data> {
    fn num_relocations(&self) -> usize {
        todo!()
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct VerneedTable<'data> {
    _phantom: PhantomData<&'data ()>,
}

impl<'data> platform::VerneedTable<'data> for VerneedTable<'data> {
    fn version_name(&self, local_symbol_index: SymbolIndex) -> Option<&'data [u8]> {
        todo!()
    }
}

pub(crate) struct RawSymbolName<'data> {
    name: &'data [u8],
}

impl<'data> Display for RawSymbolName<'data> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<'data> platform::RawSymbolName<'data> for RawSymbolName<'data> {
    fn parse(bytes: &'data [u8]) -> Self {
        todo!()
    }

    fn name(&self) -> &'data [u8] {
        todo!()
    }

    fn version_name(&self) -> Option<&'data [u8]> {
        todo!()
    }

    fn is_default(&self) -> bool {
        todo!()
    }
}

struct SectionRule<'data> {
    _phantom: PhantomData<&'data ()>,
}

struct ObjectLayoutStateExt<'data> {
    _phantom: PhantomData<&'data ()>,
}
