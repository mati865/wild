use anyhow::Result;
use object::LittleEndian;
use object::read::elf::ProgramHeader as _;
use object::read::elf::SectionHeader;
use std::borrow::Cow;
use std::fmt;

macro_rules! const_name_by_value {
    ($needle: expr, $( $const:ident ),*) => {
        match $needle {
            $(object::elf::$const => Some(stringify!($const)),)*
            _ => None
        }
    };
}

#[must_use]
pub fn x86_64_rel_type_to_string(r_type: u32) -> Cow<'static, str> {
    if let Some(name) = const_name_by_value![
        r_type,
        R_X86_64_NONE,
        R_X86_64_64,
        R_X86_64_PC32,
        R_X86_64_GOT32,
        R_X86_64_PLT32,
        R_X86_64_COPY,
        R_X86_64_GLOB_DAT,
        R_X86_64_JUMP_SLOT,
        R_X86_64_RELATIVE,
        R_X86_64_GOTPCREL,
        R_X86_64_32,
        R_X86_64_32S,
        R_X86_64_16,
        R_X86_64_PC16,
        R_X86_64_8,
        R_X86_64_PC8,
        R_X86_64_DTPMOD64,
        R_X86_64_DTPOFF64,
        R_X86_64_TPOFF64,
        R_X86_64_TLSGD,
        R_X86_64_TLSLD,
        R_X86_64_DTPOFF32,
        R_X86_64_GOTTPOFF,
        R_X86_64_TPOFF32,
        R_X86_64_PC64,
        R_X86_64_GOTOFF64,
        R_X86_64_GOTPC32,
        R_X86_64_GOT64,
        R_X86_64_GOTPCREL64,
        R_X86_64_GOTPC64,
        R_X86_64_GOTPLT64,
        R_X86_64_PLTOFF64,
        R_X86_64_SIZE32,
        R_X86_64_SIZE64,
        R_X86_64_GOTPC32_TLSDESC,
        R_X86_64_TLSDESC_CALL,
        R_X86_64_TLSDESC,
        R_X86_64_IRELATIVE,
        R_X86_64_RELATIVE64,
        R_X86_64_GOTPCRELX,
        R_X86_64_REX_GOTPCRELX
    ] {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("Unknown x86_64 relocation type 0x{r_type:x}"))
    }
}

#[must_use]
pub fn aarch64_rel_type_to_string(r_type: u32) -> Cow<'static, str> {
    if let Some(name) = const_name_by_value![
        r_type,
        R_AARCH64_NONE,
        R_AARCH64_P32_ABS32,
        R_AARCH64_P32_COPY,
        R_AARCH64_P32_GLOB_DAT,
        R_AARCH64_P32_JUMP_SLOT,
        R_AARCH64_P32_RELATIVE,
        R_AARCH64_P32_TLS_DTPMOD,
        R_AARCH64_P32_TLS_DTPREL,
        R_AARCH64_P32_TLS_TPREL,
        R_AARCH64_P32_TLSDESC,
        R_AARCH64_P32_IRELATIVE,
        R_AARCH64_ABS64,
        R_AARCH64_ABS32,
        R_AARCH64_ABS16,
        R_AARCH64_PREL64,
        R_AARCH64_PREL32,
        R_AARCH64_PREL16,
        R_AARCH64_MOVW_UABS_G0,
        R_AARCH64_MOVW_UABS_G0_NC,
        R_AARCH64_MOVW_UABS_G1,
        R_AARCH64_MOVW_UABS_G1_NC,
        R_AARCH64_MOVW_UABS_G2,
        R_AARCH64_MOVW_UABS_G2_NC,
        R_AARCH64_MOVW_UABS_G3,
        R_AARCH64_MOVW_SABS_G0,
        R_AARCH64_MOVW_SABS_G1,
        R_AARCH64_MOVW_SABS_G2,
        R_AARCH64_LD_PREL_LO19,
        R_AARCH64_ADR_PREL_LO21,
        R_AARCH64_ADR_PREL_PG_HI21,
        R_AARCH64_ADR_PREL_PG_HI21_NC,
        R_AARCH64_ADD_ABS_LO12_NC,
        R_AARCH64_LDST8_ABS_LO12_NC,
        R_AARCH64_TSTBR14,
        R_AARCH64_CONDBR19,
        R_AARCH64_JUMP26,
        R_AARCH64_CALL26,
        R_AARCH64_LDST16_ABS_LO12_NC,
        R_AARCH64_LDST32_ABS_LO12_NC,
        R_AARCH64_LDST64_ABS_LO12_NC,
        R_AARCH64_MOVW_PREL_G0,
        R_AARCH64_MOVW_PREL_G0_NC,
        R_AARCH64_MOVW_PREL_G1,
        R_AARCH64_MOVW_PREL_G1_NC,
        R_AARCH64_MOVW_PREL_G2,
        R_AARCH64_MOVW_PREL_G2_NC,
        R_AARCH64_MOVW_PREL_G3,
        R_AARCH64_LDST128_ABS_LO12_NC,
        R_AARCH64_MOVW_GOTOFF_G0,
        R_AARCH64_MOVW_GOTOFF_G0_NC,
        R_AARCH64_MOVW_GOTOFF_G1,
        R_AARCH64_MOVW_GOTOFF_G1_NC,
        R_AARCH64_MOVW_GOTOFF_G2,
        R_AARCH64_MOVW_GOTOFF_G2_NC,
        R_AARCH64_MOVW_GOTOFF_G3,
        R_AARCH64_GOTREL64,
        R_AARCH64_GOTREL32,
        R_AARCH64_GOT_LD_PREL19,
        R_AARCH64_LD64_GOTOFF_LO15,
        R_AARCH64_ADR_GOT_PAGE,
        R_AARCH64_LD64_GOT_LO12_NC,
        R_AARCH64_LD64_GOTPAGE_LO15,
        R_AARCH64_TLSGD_ADR_PREL21,
        R_AARCH64_TLSGD_ADR_PAGE21,
        R_AARCH64_TLSGD_ADD_LO12_NC,
        R_AARCH64_TLSGD_MOVW_G1,
        R_AARCH64_TLSGD_MOVW_G0_NC,
        R_AARCH64_TLSLD_ADR_PREL21,
        R_AARCH64_TLSLD_ADR_PAGE21,
        R_AARCH64_TLSLD_ADD_LO12_NC,
        R_AARCH64_TLSLD_MOVW_G1,
        R_AARCH64_TLSLD_MOVW_G0_NC,
        R_AARCH64_TLSLD_LD_PREL19,
        R_AARCH64_TLSLD_MOVW_DTPREL_G2,
        R_AARCH64_TLSLD_MOVW_DTPREL_G1,
        R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC,
        R_AARCH64_TLSLD_MOVW_DTPREL_G0,
        R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC,
        R_AARCH64_TLSLD_ADD_DTPREL_HI12,
        R_AARCH64_TLSLD_ADD_DTPREL_LO12,
        R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC,
        R_AARCH64_TLSLD_LDST8_DTPREL_LO12,
        R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC,
        R_AARCH64_TLSLD_LDST16_DTPREL_LO12,
        R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC,
        R_AARCH64_TLSLD_LDST32_DTPREL_LO12,
        R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC,
        R_AARCH64_TLSLD_LDST64_DTPREL_LO12,
        R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC,
        R_AARCH64_TLSIE_MOVW_GOTTPREL_G1,
        R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC,
        R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21,
        R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC,
        R_AARCH64_TLSIE_LD_GOTTPREL_PREL19,
        R_AARCH64_TLSLE_MOVW_TPREL_G2,
        R_AARCH64_TLSLE_MOVW_TPREL_G1,
        R_AARCH64_TLSLE_MOVW_TPREL_G1_NC,
        R_AARCH64_TLSLE_MOVW_TPREL_G0,
        R_AARCH64_TLSLE_MOVW_TPREL_G0_NC,
        R_AARCH64_TLSLE_ADD_TPREL_HI12,
        R_AARCH64_TLSLE_ADD_TPREL_LO12,
        R_AARCH64_TLSLE_ADD_TPREL_LO12_NC,
        R_AARCH64_TLSLE_LDST8_TPREL_LO12,
        R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC,
        R_AARCH64_TLSLE_LDST16_TPREL_LO12,
        R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC,
        R_AARCH64_TLSLE_LDST32_TPREL_LO12,
        R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC,
        R_AARCH64_TLSLE_LDST64_TPREL_LO12,
        R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC,
        R_AARCH64_TLSDESC_LD_PREL19,
        R_AARCH64_TLSDESC_ADR_PREL21,
        R_AARCH64_TLSDESC_ADR_PAGE21,
        R_AARCH64_TLSDESC_LD64_LO12,
        R_AARCH64_TLSDESC_ADD_LO12,
        R_AARCH64_TLSDESC_OFF_G1,
        R_AARCH64_TLSDESC_OFF_G0_NC,
        R_AARCH64_TLSDESC_LDR,
        R_AARCH64_TLSDESC_ADD,
        R_AARCH64_TLSDESC_CALL,
        R_AARCH64_TLSLE_LDST128_TPREL_LO12,
        R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC,
        R_AARCH64_TLSLD_LDST128_DTPREL_LO12,
        R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC,
        R_AARCH64_COPY,
        R_AARCH64_GLOB_DAT,
        R_AARCH64_JUMP_SLOT,
        R_AARCH64_RELATIVE,
        R_AARCH64_TLS_DTPMOD,
        R_AARCH64_TLS_DTPREL,
        R_AARCH64_TLS_TPREL,
        R_AARCH64_TLSDESC,
        R_AARCH64_IRELATIVE
    ] {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("Unknown aarch64 relocation type 0x{r_type:x}"))
    }
}

#[must_use]
pub fn riscv64_rel_type_to_string(r_type: u32) -> Cow<'static, str> {
    if let Some(name) = const_name_by_value![
        r_type,
        R_RISCV_NONE,
        R_RISCV_32,
        R_RISCV_64,
        R_RISCV_RELATIVE,
        R_RISCV_COPY,
        R_RISCV_JUMP_SLOT,
        R_RISCV_TLS_DTPMOD32,
        R_RISCV_TLS_DTPMOD64,
        R_RISCV_TLS_DTPREL32,
        R_RISCV_TLS_DTPREL64,
        R_RISCV_TLS_TPREL32,
        R_RISCV_TLS_TPREL64,
        R_RISCV_TLSDESC,
        R_RISCV_BRANCH,
        R_RISCV_JAL,
        R_RISCV_CALL,
        R_RISCV_CALL_PLT,
        R_RISCV_GOT_HI20,
        R_RISCV_TLS_GOT_HI20,
        R_RISCV_TLS_GD_HI20,
        R_RISCV_PCREL_HI20,
        R_RISCV_PCREL_LO12_I,
        R_RISCV_PCREL_LO12_S,
        R_RISCV_HI20,
        R_RISCV_LO12_I,
        R_RISCV_LO12_S,
        R_RISCV_TPREL_HI20,
        R_RISCV_TPREL_LO12_I,
        R_RISCV_TPREL_LO12_S,
        R_RISCV_TPREL_ADD,
        R_RISCV_ADD8,
        R_RISCV_ADD16,
        R_RISCV_ADD32,
        R_RISCV_ADD64,
        R_RISCV_SUB8,
        R_RISCV_SUB16,
        R_RISCV_SUB32,
        R_RISCV_SUB64,
        R_RISCV_GOT32_PCREL,
        R_RISCV_ALIGN,
        R_RISCV_RVC_BRANCH,
        R_RISCV_RVC_JUMP,
        R_RISCV_RVC_LUI,
        R_RISCV_GPREL_I,
        R_RISCV_GPREL_S,
        R_RISCV_TPREL_I,
        R_RISCV_TPREL_S,
        R_RISCV_RELAX,
        R_RISCV_SUB6,
        R_RISCV_SET6,
        R_RISCV_SET8,
        R_RISCV_SET16,
        R_RISCV_SET32,
        R_RISCV_32_PCREL,
        R_RISCV_IRELATIVE,
        R_RISCV_PLT32,
        R_RISCV_SET_ULEB128,
        R_RISCV_SUB_ULEB128,
        R_RISCV_TLSDESC_HI20,
        R_RISCV_TLSDESC_LOAD_LO12,
        R_RISCV_TLSDESC_ADD_LO12,
        R_RISCV_TLSDESC_CALL
    ] {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("Unknown riscv64 relocation type 0x{r_type:x}"))
    }
}

#[must_use]
pub fn segment_type_to_string(p_type: u32) -> Cow<'static, str> {
    if let Some(name) = const_name_by_value![
        p_type,
        PT_NULL,
        PT_LOAD,
        PT_DYNAMIC,
        PT_INTERP,
        PT_NOTE,
        PT_SHLIB,
        PT_PHDR,
        PT_TLS,
        PT_GNU_EH_FRAME,
        PT_GNU_STACK,
        PT_GNU_RELRO,
        PT_GNU_PROPERTY,
        // RISC-V specific program headers
        SHT_RISCV_ATTRIBUTES
    ] {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("UNKNOWN_P_TYPE_{p_type}"))
    }
}

/// Section flag bit values.
pub mod shf {
    use super::SectionFlags;

    pub const WRITE: SectionFlags = SectionFlags::from_u32(object::elf::SHF_WRITE);
    pub const ALLOC: SectionFlags = SectionFlags::from_u32(object::elf::SHF_ALLOC);
    pub const EXECINSTR: SectionFlags = SectionFlags::from_u32(object::elf::SHF_EXECINSTR);
    pub const MERGE: SectionFlags = SectionFlags::from_u32(object::elf::SHF_MERGE);
    pub const STRINGS: SectionFlags = SectionFlags::from_u32(object::elf::SHF_STRINGS);
    pub const INFO_LINK: SectionFlags = SectionFlags::from_u32(object::elf::SHF_INFO_LINK);
    pub const LINK_ORDER: SectionFlags = SectionFlags::from_u32(object::elf::SHF_LINK_ORDER);
    pub const OS_NONCONFORMING: SectionFlags =
        SectionFlags::from_u32(object::elf::SHF_OS_NONCONFORMING);
    pub const GROUP: SectionFlags = SectionFlags::from_u32(object::elf::SHF_GROUP);
    pub const TLS: SectionFlags = SectionFlags::from_u32(object::elf::SHF_TLS);
    pub const COMPRESSED: SectionFlags = SectionFlags::from_u32(object::elf::SHF_COMPRESSED);
    pub const GNU_RETAIN: SectionFlags = SectionFlags::from_u32(object::elf::SHF_GNU_RETAIN);
}

pub mod sht {
    use super::SectionType;

    pub const NULL: SectionType = SectionType(object::elf::SHT_NULL);
    pub const PROGBITS: SectionType = SectionType(object::elf::SHT_PROGBITS);
    pub const SYMTAB: SectionType = SectionType(object::elf::SHT_SYMTAB);
    pub const STRTAB: SectionType = SectionType(object::elf::SHT_STRTAB);
    pub const RELA: SectionType = SectionType(object::elf::SHT_RELA);
    pub const HASH: SectionType = SectionType(object::elf::SHT_HASH);
    pub const DYNAMIC: SectionType = SectionType(object::elf::SHT_DYNAMIC);
    pub const NOTE: SectionType = SectionType(object::elf::SHT_NOTE);
    pub const NOBITS: SectionType = SectionType(object::elf::SHT_NOBITS);
    pub const REL: SectionType = SectionType(object::elf::SHT_REL);
    pub const SHLIB: SectionType = SectionType(object::elf::SHT_SHLIB);
    pub const DYNSYM: SectionType = SectionType(object::elf::SHT_DYNSYM);
    pub const INIT_ARRAY: SectionType = SectionType(object::elf::SHT_INIT_ARRAY);
    pub const FINI_ARRAY: SectionType = SectionType(object::elf::SHT_FINI_ARRAY);
    pub const PREINIT_ARRAY: SectionType = SectionType(object::elf::SHT_PREINIT_ARRAY);
    pub const GROUP: SectionType = SectionType(object::elf::SHT_GROUP);
    pub const SYMTAB_SHNDX: SectionType = SectionType(object::elf::SHT_SYMTAB_SHNDX);
    pub const LOOS: SectionType = SectionType(object::elf::SHT_LOOS);
    pub const GNU_ATTRIBUTES: SectionType = SectionType(object::elf::SHT_GNU_ATTRIBUTES);
    pub const GNU_HASH: SectionType = SectionType(object::elf::SHT_GNU_HASH);
    pub const GNU_LIBLIST: SectionType = SectionType(object::elf::SHT_GNU_LIBLIST);
    pub const CHECKSUM: SectionType = SectionType(object::elf::SHT_CHECKSUM);
    pub const LOSUNW: SectionType = SectionType(object::elf::SHT_LOSUNW);
    pub const SUNW_COMDAT: SectionType = SectionType(object::elf::SHT_SUNW_COMDAT);
    pub const SUNW_SYMINFO: SectionType = SectionType(object::elf::SHT_SUNW_syminfo);
    pub const GNU_VERDEF: SectionType = SectionType(object::elf::SHT_GNU_VERDEF);
    pub const GNU_VERNEED: SectionType = SectionType(object::elf::SHT_GNU_VERNEED);
    pub const GNU_VERSYM: SectionType = SectionType(object::elf::SHT_GNU_VERSYM);
    pub const HISUNW: SectionType = SectionType(object::elf::SHT_HISUNW);
    pub const HIOS: SectionType = SectionType(object::elf::SHT_HIOS);
    pub const LOPROC: SectionType = SectionType(object::elf::SHT_LOPROC);
    pub const HIPROC: SectionType = SectionType(object::elf::SHT_HIPROC);
    pub const LOUSER: SectionType = SectionType(object::elf::SHT_LOUSER);
    pub const HIUSER: SectionType = SectionType(object::elf::SHT_HIUSER);
    pub const RISCV_ATTRIBUTES: SectionType = SectionType(object::elf::SHT_RISCV_ATTRIBUTES);
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SectionFlags(u32);

impl SectionFlags {
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub fn from_header(header: &object::elf::SectionHeader64<LittleEndian>) -> Self {
        Self(header.sh_flags(LittleEndian) as u32)
    }

    #[must_use]
    pub fn contains(self, flag: SectionFlags) -> bool {
        self.0 & flag.0 != 0
    }

    #[must_use]
    pub const fn from_u32(raw: u32) -> SectionFlags {
        SectionFlags(raw)
    }

    /// Returns self with the specified flags set.
    #[must_use]
    pub const fn with(self, flags: SectionFlags) -> SectionFlags {
        SectionFlags(self.0 | flags.0)
    }

    /// Returns self with the specified flags cleared.
    #[must_use]
    pub const fn without(self, flags: SectionFlags) -> SectionFlags {
        SectionFlags(self.0 & !flags.0)
    }

    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0 as u64
    }

    #[must_use]
    pub fn should_retain(&self) -> bool {
        self.contains(shf::GNU_RETAIN)
    }
}

impl From<u64> for SectionFlags {
    fn from(value: u64) -> Self {
        Self(value as u32)
    }
}

impl std::fmt::Display for SectionFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (flag, ch) in [
            (shf::WRITE, "W"),
            (shf::ALLOC, "A"),
            (shf::EXECINSTR, "X"),
            (shf::MERGE, "M"),
            (shf::STRINGS, "S"),
            (shf::INFO_LINK, "I"),
            (shf::LINK_ORDER, "L"),
            (shf::OS_NONCONFORMING, "O"),
            (shf::GROUP, "G"),
            (shf::TLS, "T"),
            (shf::COMPRESSED, "C"),
            // TODO: ld linker sometimes propagates the flag
            // (shf::GNU_RETAIN, "R"),
        ] {
            if self.contains(flag) {
                f.write_str(ch)?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for SectionFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self, f)
    }
}

impl std::ops::BitOrAssign for SectionFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for SectionFlags {
    type Output = SectionFlags;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SectionType(u32);

impl SectionType {
    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub fn from_header(header: &object::elf::SectionHeader64<LittleEndian>) -> Self {
        Self(header.sh_type(LittleEndian))
    }

    #[must_use]
    pub fn from_u32(raw: u32) -> Self {
        Self(raw)
    }
}

pub mod secnames {
    pub const FILEHEADER_SECTION_NAME_STR: &str = "";
    pub const FILEHEADER_SECTION_NAME: &[u8] = FILEHEADER_SECTION_NAME_STR.as_bytes();
    pub const RODATA_SECTION_NAME_STR: &str = ".rodata";
    pub const RODATA_SECTION_NAME: &[u8] = RODATA_SECTION_NAME_STR.as_bytes();
    pub const TEXT_SECTION_NAME_STR: &str = ".text";
    pub const TEXT_SECTION_NAME: &[u8] = TEXT_SECTION_NAME_STR.as_bytes();
    pub const INIT_ARRAY_SECTION_NAME_STR: &str = ".init_array";
    pub const INIT_ARRAY_SECTION_NAME: &[u8] = INIT_ARRAY_SECTION_NAME_STR.as_bytes();
    pub const FINI_ARRAY_SECTION_NAME_STR: &str = ".fini_array";
    pub const FINI_ARRAY_SECTION_NAME: &[u8] = FINI_ARRAY_SECTION_NAME_STR.as_bytes();
    pub const PREINIT_ARRAY_SECTION_NAME_STR: &str = ".preinit_array";
    pub const PREINIT_ARRAY_SECTION_NAME: &[u8] = PREINIT_ARRAY_SECTION_NAME_STR.as_bytes();
    pub const DATA_SECTION_NAME_STR: &str = ".data";
    pub const DATA_SECTION_NAME: &[u8] = DATA_SECTION_NAME_STR.as_bytes();
    pub const EH_FRAME_SECTION_NAME_STR: &str = ".eh_frame";
    pub const EH_FRAME_SECTION_NAME: &[u8] = EH_FRAME_SECTION_NAME_STR.as_bytes();
    pub const EH_FRAME_HDR_SECTION_NAME_STR: &str = ".eh_frame_hdr";
    pub const EH_FRAME_HDR_SECTION_NAME: &[u8] = EH_FRAME_HDR_SECTION_NAME_STR.as_bytes();
    pub const SHSTRTAB_SECTION_NAME_STR: &str = ".shstrtab";
    pub const SHSTRTAB_SECTION_NAME: &[u8] = SHSTRTAB_SECTION_NAME_STR.as_bytes();
    pub const SYMTAB_SECTION_NAME_STR: &str = ".symtab";
    pub const SYMTAB_SECTION_NAME: &[u8] = SYMTAB_SECTION_NAME_STR.as_bytes();
    pub const STRTAB_SECTION_NAME_STR: &str = ".strtab";
    pub const STRTAB_SECTION_NAME: &[u8] = STRTAB_SECTION_NAME_STR.as_bytes();
    pub const TDATA_SECTION_NAME_STR: &str = ".tdata";
    pub const TDATA_SECTION_NAME: &[u8] = TDATA_SECTION_NAME_STR.as_bytes();
    pub const TBSS_SECTION_NAME_STR: &str = ".tbss";
    pub const TBSS_SECTION_NAME: &[u8] = TBSS_SECTION_NAME_STR.as_bytes();
    pub const BSS_SECTION_NAME_STR: &str = ".bss";
    pub const BSS_SECTION_NAME: &[u8] = BSS_SECTION_NAME_STR.as_bytes();
    pub const GOT_SECTION_NAME_STR: &str = ".got";
    pub const GOT_SECTION_NAME: &[u8] = GOT_SECTION_NAME_STR.as_bytes();
    pub const INIT_SECTION_NAME_STR: &str = ".init";
    pub const INIT_SECTION_NAME: &[u8] = INIT_SECTION_NAME_STR.as_bytes();
    pub const FINI_SECTION_NAME_STR: &str = ".fini";
    pub const FINI_SECTION_NAME: &[u8] = FINI_SECTION_NAME_STR.as_bytes();
    pub const RELA_PLT_SECTION_NAME_STR: &str = ".rela.plt";
    pub const RELA_PLT_SECTION_NAME: &[u8] = RELA_PLT_SECTION_NAME_STR.as_bytes();
    pub const COMMENT_SECTION_NAME_STR: &str = ".comment";
    pub const COMMENT_SECTION_NAME: &[u8] = COMMENT_SECTION_NAME_STR.as_bytes();
    pub const DYNAMIC_SECTION_NAME_STR: &str = ".dynamic";
    pub const DYNAMIC_SECTION_NAME: &[u8] = DYNAMIC_SECTION_NAME_STR.as_bytes();
    pub const DYNSYM_SECTION_NAME_STR: &str = ".dynsym";
    pub const DYNSYM_SECTION_NAME: &[u8] = DYNSYM_SECTION_NAME_STR.as_bytes();
    pub const DYNSTR_SECTION_NAME_STR: &str = ".dynstr";
    pub const DYNSTR_SECTION_NAME: &[u8] = DYNSTR_SECTION_NAME_STR.as_bytes();
    pub const RELA_DYN_SECTION_NAME_STR: &str = ".rela.dyn";
    pub const RELA_DYN_SECTION_NAME: &[u8] = RELA_DYN_SECTION_NAME_STR.as_bytes();
    pub const GCC_EXCEPT_TABLE_SECTION_NAME_STR: &str = ".gcc_except_table";
    pub const GCC_EXCEPT_TABLE_SECTION_NAME: &[u8] = GCC_EXCEPT_TABLE_SECTION_NAME_STR.as_bytes();
    pub const INTERP_SECTION_NAME_STR: &str = ".interp";
    pub const INTERP_SECTION_NAME: &[u8] = INTERP_SECTION_NAME_STR.as_bytes();
    pub const GNU_VERSION_SECTION_NAME_STR: &str = ".gnu.version";
    pub const GNU_VERSION_SECTION_NAME: &[u8] = GNU_VERSION_SECTION_NAME_STR.as_bytes();
    pub const GNU_VERSION_D_SECTION_NAME_STR: &str = ".gnu.version_d";
    pub const GNU_VERSION_D_SECTION_NAME: &[u8] = GNU_VERSION_D_SECTION_NAME_STR.as_bytes();
    pub const GNU_VERSION_R_SECTION_NAME_STR: &str = ".gnu.version_r";
    pub const GNU_VERSION_R_SECTION_NAME: &[u8] = GNU_VERSION_R_SECTION_NAME_STR.as_bytes();
    pub const PROGRAM_HEADERS_SECTION_NAME_STR: &str = ".phdr";
    pub const PROGRAM_HEADERS_SECTION_NAME: &[u8] = PROGRAM_HEADERS_SECTION_NAME_STR.as_bytes();
    pub const SECTION_HEADERS_SECTION_NAME_STR: &str = ".shdr";
    pub const SECTION_HEADERS_SECTION_NAME: &[u8] = SECTION_HEADERS_SECTION_NAME_STR.as_bytes();
    pub const GNU_HASH_SECTION_NAME_STR: &str = ".gnu.hash";
    pub const GNU_HASH_SECTION_NAME: &[u8] = GNU_HASH_SECTION_NAME_STR.as_bytes();
    pub const PLT_SECTION_NAME_STR: &str = ".plt";
    pub const PLT_SECTION_NAME: &[u8] = PLT_SECTION_NAME_STR.as_bytes();
    pub const IPLT_SECTION_NAME_STR: &str = ".iplt";
    pub const IPLT_SECTION_NAME: &[u8] = IPLT_SECTION_NAME_STR.as_bytes();
    pub const PLT_GOT_SECTION_NAME_STR: &str = ".plt.got";
    pub const PLT_GOT_SECTION_NAME: &[u8] = PLT_GOT_SECTION_NAME_STR.as_bytes();
    pub const GOT_PLT_SECTION_NAME_STR: &str = ".got.plt";
    pub const GOT_PLT_SECTION_NAME: &[u8] = GOT_PLT_SECTION_NAME_STR.as_bytes();
    pub const PLT_SEC_SECTION_NAME_STR: &str = ".plt.sec";
    pub const PLT_SEC_SECTION_NAME: &[u8] = PLT_SEC_SECTION_NAME_STR.as_bytes();
    pub const NOTE_ABI_TAG_SECTION_NAME_STR: &str = ".note.ABI-tag";
    pub const NOTE_ABI_TAG_SECTION_NAME: &[u8] = NOTE_ABI_TAG_SECTION_NAME_STR.as_bytes();
    pub const NOTE_GNU_PROPERTY_SECTION_NAME_STR: &str = ".note.gnu.property";
    pub const NOTE_GNU_PROPERTY_SECTION_NAME: &[u8] = NOTE_GNU_PROPERTY_SECTION_NAME_STR.as_bytes();
    pub const NOTE_GNU_BUILD_ID_SECTION_NAME_STR: &str = ".note.gnu.build-id";
    pub const NOTE_GNU_BUILD_ID_SECTION_NAME: &[u8] = NOTE_GNU_BUILD_ID_SECTION_NAME_STR.as_bytes();
    pub const DEBUG_LOC_SECTION_NAME_STR: &str = ".debug_loc";
    pub const DEBUG_LOC_SECTION_NAME: &[u8] = DEBUG_LOC_SECTION_NAME_STR.as_bytes();
    pub const DEBUG_RANGES_SECTION_NAME_STR: &str = ".debug_ranges";
    pub const DEBUG_RANGES_SECTION_NAME: &[u8] = DEBUG_RANGES_SECTION_NAME_STR.as_bytes();
    pub const GROUP_SECTION_NAME_STR: &str = ".group";
    pub const GROUP_SECTION_NAME: &[u8] = GROUP_SECTION_NAME_STR.as_bytes();
    pub const DATA_REL_RO_SECTION_NAME_STR: &str = ".data.rel.ro";
    pub const DATA_REL_RO_SECTION_NAME: &[u8] = DATA_REL_RO_SECTION_NAME_STR.as_bytes();
    pub const RISCV_ATTRIBUTES_SECTION_NAME_STR: &str = ".riscv.attributes";
    pub const RISCV_ATTRIBUTES_SECTION_NAME: &[u8] = RISCV_ATTRIBUTES_SECTION_NAME_STR.as_bytes();

    pub const GNU_LTO_SYMTAB_PREFIX: &str = ".gnu.lto_.symtab";
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SegmentType(u32);

impl SegmentType {
    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub fn from_header(header: &object::elf::ProgramHeader64<LittleEndian>) -> Self {
        Self(header.p_type(LittleEndian))
    }

    #[must_use]
    pub const fn from_u32(raw: u32) -> Self {
        Self(raw)
    }
}

impl std::fmt::Display for SegmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            pt::PHDR => write!(f, "PHDR")?,
            pt::INTERP => write!(f, "INTERP")?,
            pt::NOTE => write!(f, "NOTE")?,
            pt::LOAD => write!(f, "LOAD")?,
            pt::TLS => write!(f, "TLS")?,
            pt::GNU_EH_FRAME => write!(f, "GNU_EH_FRAME")?,
            pt::DYNAMIC => write!(f, "DYNAMIC")?,
            pt::GNU_RELRO => write!(f, "GNU_RELRO")?,
            pt::GNU_STACK => write!(f, "GNU_STACK")?,
            other => write!(f, "UNKNOWN_SEG_TYPE({})", other.raw())?,
        }
        Ok(())
    }
}

pub mod pt {
    use super::SegmentType;

    pub const PHDR: SegmentType = SegmentType::from_u32(object::elf::PT_PHDR);
    pub const INTERP: SegmentType = SegmentType::from_u32(object::elf::PT_INTERP);
    pub const NOTE: SegmentType = SegmentType::from_u32(object::elf::PT_NOTE);
    pub const LOAD: SegmentType = SegmentType::from_u32(object::elf::PT_LOAD);
    pub const TLS: SegmentType = SegmentType::from_u32(object::elf::PT_TLS);
    pub const GNU_EH_FRAME: SegmentType = SegmentType::from_u32(object::elf::PT_GNU_EH_FRAME);
    pub const DYNAMIC: SegmentType = SegmentType::from_u32(object::elf::PT_DYNAMIC);
    pub const GNU_RELRO: SegmentType = SegmentType::from_u32(object::elf::PT_GNU_RELRO);
    pub const GNU_STACK: SegmentType = SegmentType::from_u32(object::elf::PT_GNU_STACK);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SegmentFlags(u32);

impl SegmentFlags {
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub fn from_header(header: &object::elf::ProgramHeader64<LittleEndian>) -> Self {
        Self(header.p_flags(LittleEndian))
    }

    #[must_use]
    pub fn contains(self, flag: SegmentFlags) -> bool {
        self.0 & flag.0 != 0
    }

    #[must_use]
    pub const fn from_u32(raw: u32) -> SegmentFlags {
        SegmentFlags(raw)
    }

    /// Returns self with the specified flags set.
    #[must_use]
    pub const fn with(self, flags: SegmentFlags) -> SegmentFlags {
        SegmentFlags(self.0 | flags.0)
    }

    /// Returns self with the specified flags cleared.
    #[must_use]
    pub const fn without(self, flags: SegmentFlags) -> SegmentFlags {
        SegmentFlags(self.0 & !flags.0)
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

pub mod pf {
    use super::SegmentFlags;

    pub const EXECUTABLE: SegmentFlags = SegmentFlags::from_u32(object::elf::PF_X);
    pub const WRITABLE: SegmentFlags = SegmentFlags::from_u32(object::elf::PF_W);
    pub const READABLE: SegmentFlags = SegmentFlags::from_u32(object::elf::PF_R);
}

impl std::fmt::Display for SegmentFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.contains(pf::WRITABLE) {
            f.write_str("W")?;
        }
        if self.contains(pf::READABLE) {
            f.write_str("R")?;
        }
        if self.contains(pf::EXECUTABLE) {
            f.write_str("X")?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for SegmentFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self, f)
    }
}

impl std::ops::BitOrAssign for SegmentFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for SegmentFlags {
    type Output = SegmentFlags;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

// RISC-V related constants

/// Dynamic thread vector pointers point 0x800 past the start of each TLS block.
pub const RISCV_TLS_DTV_OFFSET: u64 = 0x800;

pub const RISCV_ATTRIBUTE_VENDOR_NAME: &str = "riscv";

// RISC-V ELF Tag constants, see: https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#risc-v-specific-dynamic-section-tags
pub mod riscvattr {
    // Attributes relate to whole file.
    pub const TAG_RISCV_WHOLE_FILE: u64 = 1;
    // Indicates the stack alignment requirement in bytes (ULEB128).
    pub const TAG_RISCV_STACK_ALIGN: u64 = 4;
    // Indicates the target architecture of this object (NTBS).
    pub const TAG_RISCV_ARCH: u64 = 5;
    // Indicates whether to impose unaligned memory accesses in code generation (ULEB128).
    pub const TAG_RISCV_UNALIGNED_ACCESS: u64 = 6;
    // Indicates the major version of the privileged specification (ULEB128, deprecated).
    pub const TAG_RISCV_PRIV_SPEC: u64 = 8;
    // Indicates the minor version of the privileged specification (ULEB128, deprecated).
    pub const TAG_RISCV_PRIV_SPEC_MINOR: u64 = 10;
    // Indicates the revision version of the privileged specification (ULEB128, deprecated).
    pub const TAG_RISCV_PRIV_SPEC_REVISION: u64 = 12;
    // Indicates which version of the atomics ABI is being used (ULEB128).
    pub const TAG_RISCV_ATOMIC_ABI: u64 = 14;
    // Indicates the usage definition of the X3 register (ULEB128).
    pub const TAG_RISCV_X3_REG_USAGE: u64 = 16;
}

/// For additional information on ELF relocation types, see "ELF-64 Object File Format" -
/// https://uclibc.org/docs/elf-64-gen.pdf. For information on the TLS related relocations, see "ELF
/// Handling For Thread-Local Storage" - https://www.uclibc.org/docs/tls.pdf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocationKind {
    /// The absolute address of a symbol or section.
    Absolute,

    /// The absolute address of a symbol or section related to EH section.
    AbsoluteSet,

    /// The 6 low bits of an absolute address of a symbol or section.
    AbsoluteSetWord6,

    /// Add the absolute address of a symbol or section at the place of the relocation
    /// to the value at the place.
    AbsoluteAddition,

    /// Subtract the absolute address of a symbol or section at the place of the relocation
    /// from the value at the place.
    AbsoluteSubtraction,

    /// Subtract the absolute address of a symbol or section at the place of the relocation
    /// from the value at the place (use WORD6 type for the operation)
    AbsoluteSubtractionWord6,

    /// The absolute address of a symbol or section. We are going to extract only the offset
    /// within a page, so dynamic relocation creation must be skipped.
    AbsoluteAArch64,

    /// Subtract addresses of two symbols and encode the value using ULEB128.
    PairSubtraction,

    /// The address of the symbol, relative to the place of the relocation.
    Relative,

    /// The address of the symbol, relative to the place of the relocation. The address of the relocation
    /// points to an instruction for which the R_RISCV_PCREL_HI20 relocation is used and that is the place
    /// we make this relocation relative to.
    RelativeRiscVLow12,

    /// The address of the symbol, relative to the base address of the GOT.
    SymRelGotBase,

    /// The offset of the symbol's GOT entry, relative to the start of the GOT.
    GotRelGotBase,

    /// The address of the symbol's GOT entry.
    Got,

    /// The address of the symbol's PLT entry, relative to the base address of the GOT.
    PltRelGotBase,

    /// The address of the symbol's PLT entry, relative to the place of relocation.
    PltRelative,

    /// The address of the symbol's GOT entry, relative to the place of the relocation.
    GotRelative,

    /// The address of a TLSGD structure, relative to the place of the relocation. A TLSGD
    /// (thread-local storage general dynamic) structure is a pair of values containing a module ID
    /// and the offset within that module's TLS storage.
    TlsGd,

    /// The address of the symbol's TLSGD GOT entry.
    TlsGdGot,

    /// The address of the symbol's TLSGD GOT entry, relative to the start of the GOT.
    TlsGdGotBase,

    /// The address of the TLS module ID for the shared object that we're writing, relative to the
    /// place of the relocation. This is used when a TLS variable is defined and used within the
    /// same shared object.
    TlsLd,

    /// The address of the TLS module ID for the shared object that we're writing.
    TlsLdGot,

    /// The address of the TLS module ID for the shared object that we're writing,
    /// relative to the start of the GOT.
    TlsLdGotBase,

    /// The offset of a thread-local within the TLS storage of DSO that defines that thread-local.
    DtpOff,

    /// The address of a GOT entry containing the offset of a TLS variable within the executable's
    /// TLS storage, relative to the place of the relocation.
    GotTpOff,

    /// The address of a GOT entry containing the offset of a TLS variable within the executable's
    /// TLS storage.
    GotTpOffGot,

    /// The address of a GOT entry containing the offset of a TLS variable within the executable's
    /// TLS storage, relative to the start of the GOT.
    GotTpOffGotBase,

    /// The offset of a TLS variable within the executable's TLS storage.
    TpOff,

    /// The address of a TLS descriptor structure, relative to the place of the relocation.
    TlsDesc,

    /// The address of a TLS descriptor structure.
    TlsDescGot,

    /// The address of a TLS descriptor structure, relative to the start of the GOT.
    TlsDescGotBase,

    /// Call to the TLS descriptor trampoline. Used only as a placeholder for a linker relaxation opportunity.
    TlsDescCall,

    /// No relocation needs to be applied. Produced when we eliminate a relocation due to an
    /// optimisation.
    None,

    /// The address must fulfill the alignment requirement.
    Alignment,
}

impl RelocationKind {
    #[must_use]
    pub fn is_tls(self) -> bool {
        matches!(
            self,
            Self::DtpOff
                | Self::GotTpOff
                | Self::GotTpOffGotBase
                | Self::TlsDesc
                | Self::TlsDescCall
                | Self::TlsDescGot
                | Self::TlsDescGotBase
                | Self::TlsGd
                | Self::TlsGdGot
                | Self::TlsGdGotBase
                | Self::TlsLd
                | Self::TlsLdGot
                | Self::TlsLdGotBase
                | Self::TpOff
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DynamicRelocationKind {
    Copy,
    Irelative,
    DtpMod,
    DtpOff,
    TlsDesc,
    TpOff,
    Relative,
    Absolute,
    GotEntry,
    JumpSlot,
}

impl DynamicRelocationKind {
    #[must_use]
    pub fn from_x86_64_r_type(r_type: u32) -> Option<Self> {
        let kind = match r_type {
            object::elf::R_X86_64_COPY => DynamicRelocationKind::Copy,
            object::elf::R_X86_64_IRELATIVE => DynamicRelocationKind::Irelative,
            object::elf::R_X86_64_DTPMOD64 => DynamicRelocationKind::DtpMod,
            object::elf::R_X86_64_DTPOFF64 => DynamicRelocationKind::DtpOff,
            object::elf::R_X86_64_TPOFF64 => DynamicRelocationKind::TpOff,
            object::elf::R_X86_64_RELATIVE => DynamicRelocationKind::Relative,
            object::elf::R_X86_64_GLOB_DAT => DynamicRelocationKind::GotEntry,
            object::elf::R_X86_64_64 => DynamicRelocationKind::Absolute,
            object::elf::R_X86_64_TLSDESC => DynamicRelocationKind::TlsDesc,
            object::elf::R_X86_64_JUMP_SLOT => DynamicRelocationKind::JumpSlot,
            _ => return None,
        };

        Some(kind)
    }

    #[must_use]
    pub fn x86_64_r_type(self) -> u32 {
        match self {
            DynamicRelocationKind::Copy => object::elf::R_X86_64_COPY,
            DynamicRelocationKind::Irelative => object::elf::R_X86_64_IRELATIVE,
            DynamicRelocationKind::DtpMod => object::elf::R_X86_64_DTPMOD64,
            DynamicRelocationKind::DtpOff => object::elf::R_X86_64_DTPOFF64,
            DynamicRelocationKind::TpOff => object::elf::R_X86_64_TPOFF64,
            DynamicRelocationKind::Relative => object::elf::R_X86_64_RELATIVE,
            DynamicRelocationKind::Absolute => object::elf::R_X86_64_64,
            DynamicRelocationKind::GotEntry => object::elf::R_X86_64_GLOB_DAT,
            DynamicRelocationKind::TlsDesc => object::elf::R_X86_64_TLSDESC,
            DynamicRelocationKind::JumpSlot => object::elf::R_X86_64_JUMP_SLOT,
        }
    }

    #[must_use]
    pub fn from_aarch64_r_type(r_type: u32) -> Option<Self> {
        let kind = match r_type {
            object::elf::R_AARCH64_COPY => DynamicRelocationKind::Copy,
            object::elf::R_AARCH64_IRELATIVE => DynamicRelocationKind::Irelative,
            object::elf::R_AARCH64_TLS_DTPMOD => DynamicRelocationKind::DtpMod,
            object::elf::R_AARCH64_TLS_DTPREL => DynamicRelocationKind::DtpOff,
            object::elf::R_AARCH64_TLS_TPREL => DynamicRelocationKind::TpOff,
            object::elf::R_AARCH64_RELATIVE => DynamicRelocationKind::Relative,
            object::elf::R_AARCH64_ABS64 => DynamicRelocationKind::Absolute,
            object::elf::R_AARCH64_GLOB_DAT => DynamicRelocationKind::GotEntry,
            object::elf::R_AARCH64_TLSDESC => DynamicRelocationKind::TlsDesc,
            object::elf::R_AARCH64_JUMP_SLOT => DynamicRelocationKind::JumpSlot,
            _ => return None,
        };

        Some(kind)
    }

    #[must_use]
    pub fn aarch64_r_type(&self) -> u32 {
        match self {
            DynamicRelocationKind::Copy => object::elf::R_AARCH64_COPY,
            DynamicRelocationKind::Irelative => object::elf::R_AARCH64_IRELATIVE,
            DynamicRelocationKind::DtpMod => object::elf::R_AARCH64_TLS_DTPMOD,
            DynamicRelocationKind::DtpOff => object::elf::R_AARCH64_TLS_DTPREL,
            DynamicRelocationKind::TpOff => object::elf::R_AARCH64_TLS_TPREL,
            DynamicRelocationKind::Relative => object::elf::R_AARCH64_RELATIVE,
            DynamicRelocationKind::Absolute => object::elf::R_AARCH64_ABS64,
            DynamicRelocationKind::GotEntry => object::elf::R_AARCH64_GLOB_DAT,
            DynamicRelocationKind::TlsDesc => object::elf::R_AARCH64_TLSDESC,
            DynamicRelocationKind::JumpSlot => object::elf::R_AARCH64_JUMP_SLOT,
        }
    }

    #[must_use]
    pub fn riscv64_r_type(&self) -> u32 {
        match self {
            DynamicRelocationKind::Copy => object::elf::R_RISCV_COPY,
            DynamicRelocationKind::Irelative => object::elf::R_RISCV_IRELATIVE,
            DynamicRelocationKind::DtpMod => object::elf::R_RISCV_TLS_DTPMOD64,
            DynamicRelocationKind::DtpOff => object::elf::R_RISCV_TLS_DTPREL64,
            DynamicRelocationKind::TpOff => object::elf::R_RISCV_TLS_TPREL64,
            DynamicRelocationKind::Relative => object::elf::R_RISCV_RELATIVE,
            DynamicRelocationKind::Absolute => object::elf::R_RISCV_64,
            DynamicRelocationKind::GotEntry => object::elf::R_RISCV_64,
            DynamicRelocationKind::TlsDesc => object::elf::R_RISCV_TLSDESC,
            DynamicRelocationKind::JumpSlot => object::elf::R_RISCV_JUMP_SLOT,
        }
    }
}

// Half-opened range bounded inclusively below and exclusively above: [`start`, `end`)
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub struct BitRange {
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum AArch64Instruction {
    Adr,
    Movkz,
    Movnz,
    Ldr,
    LdrRegister,
    Add,
    LdSt,
    TstBr,
    Bcond,
    JumpCall,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum RiscVInstruction {
    // The relocation encoding actually modifies the consecutive pair of instructions:
    //   10:	00000097          	auipc	ra,0x0	10: R_RISCV_CALL_PLT	symbol_name
    //   14:	000080e7          	jalr	ra # 10 <main+0x10>
    //
    // That makes the relocation pretty unusual as one would expect 2 relocations:
    // https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#procedure-calls
    UiType,

    // Encodes high 20 bits of 32-bit value and encodes the bits to upper part.
    UType,

    // Encodes low 12 bits of 32-bit value and encodes the bits to upper part.
    IType,

    // Encodes 12 bits of 32-bit value.
    SType,

    // The X-type instruction immediate encoding is defined here:
    // https://riscv.github.io/riscv-isa-manual/snapshot/unprivileged/#_immediate_encoding_variants

    // Specifies a field as the immediate field in a B-type (branch) instruction
    BType,

    // Specifies a field as the immediate field in a J-type (jump) instruction
    JType,

    // Specifies a field as the immediate field in a CB-type (compressed branch) instruction
    // https://riscv.github.io/riscv-isa-manual/snapshot/unprivileged/#_control_transfer_instructions_2
    CbType,

    // Specifies a field as the immediate field in a CJ-type (compressed jump) instruction
    CjType,

    // Encode the value using ULEB128 encoding (the size of the output is variable based on the value)
    Uleb128,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum RelocationInstruction {
    AArch64(AArch64Instruction),
    RiscV(RiscVInstruction),
}

impl RelocationInstruction {
    #[must_use]
    pub fn bit_mask(&self, range: BitRange) -> [u8; 4] {
        let mut mask = [0; 4];

        // To figure out which bits are part of the relocation, we write a value with
        // all ones into a buffer that initially contains zeros.
        let all_ones = (1 << (range.end - range.start)) - 1;
        self.write_to_value(all_ones, false, &mut mask);

        // Wherever we get a 1 is part of the relocation, so invert all bits.
        for b in &mut mask {
            *b = !*b;
        }

        mask
    }

    pub fn write_to_value(self, extracted_value: u64, negative: bool, dest: &mut [u8]) {
        match self {
            Self::AArch64(insn) => insn.write_to_value(extracted_value, negative, dest),
            Self::RiscV(insn) => insn.write_to_value(extracted_value, negative, dest),
        }
    }

    /// The inverse of `write_to_value`. Returns `(extracted_value, negative)`. Supplied `bytes`
    /// must be at least 4 bytes, otherwise we panic.
    #[must_use]
    pub fn read_value(self, bytes: &[u8]) -> (u64, bool) {
        match self {
            Self::AArch64(insn) => insn.read_value(bytes),
            Self::RiscV(insn) => insn.read_value(bytes),
        }
    }

    /// The number of bytes the relocation actually can modify in the output data.
    #[must_use]
    pub fn write_windows_size(self) -> usize {
        match self {
            Self::AArch64(..) => 4,
            Self::RiscV(..) => 10,
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum RelocationSize {
    ByteSize(usize),
    BitMasking(BitMask),
}

impl fmt::Display for RelocationSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ByteSize(bytes) => f.write_fmt(format_args!("{bytes}B")),
            Self::BitMasking(mask) => {
                f.write_fmt(format_args!("{}..{}", mask.range.start, mask.range.end))
            }
        }
    }
}

impl RelocationSize {
    pub(crate) const fn bit_mask_aarch64(
        bit_start: u32,
        bit_end: u32,
        instruction: AArch64Instruction,
    ) -> RelocationSize {
        Self::BitMasking(BitMask::new(
            RelocationInstruction::AArch64(instruction),
            bit_start,
            bit_end,
        ))
    }

    pub(crate) const fn bit_mask_riscv(
        bit_start: u32,
        bit_end: u32,
        instruction: RiscVInstruction,
    ) -> RelocationSize {
        Self::BitMasking(BitMask::new(
            RelocationInstruction::RiscV(instruction),
            bit_start,
            bit_end,
        ))
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub struct BitMask {
    pub instruction: RelocationInstruction,
    pub range: BitRange,
}

#[derive(Debug, Clone, Copy)]
pub enum PageMask {
    SymbolPlusAddendAndPosition,
    GotEntryAndPosition,
    GotBase,
}

// Allow range (half-open) of a computed value of a relocation
#[derive(Clone, Debug, Copy)]
pub struct AllowedRange {
    pub min: i64,
    pub max: i64,
}

impl AllowedRange {
    #[must_use]
    pub const fn new(min: i64, max: i64) -> Self {
        Self { min, max }
    }

    #[must_use]
    pub const fn no_check() -> Self {
        Self::new(i64::MIN, i64::MAX)
    }
}

#[derive(Clone, Debug, Copy)]
pub struct RelocationKindInfo {
    pub kind: RelocationKind,
    pub size: RelocationSize,
    pub mask: Option<PageMask>,
    pub range: AllowedRange,
    pub alignment: usize,
}

impl RelocationKindInfo {
    #[inline(always)]
    pub fn verify(&self, value: i64) -> Result<()> {
        anyhow::ensure!(
            (value as usize) & (self.alignment - 1) == 0,
            "Relocation {value} not aligned to {} bytes",
            self.alignment
        );
        anyhow::ensure!(
            self.range.min <= value && value < self.range.max,
            format!(
                "Relocation {value} outside of bounds [{}, {})",
                self.range.min, self.range.max
            )
        );
        Ok(())
    }
}

impl BitMask {
    #[must_use]
    pub const fn new(instruction: RelocationInstruction, bit_start: u32, bit_end: u32) -> Self {
        Self {
            instruction,
            range: BitRange {
                start: bit_start,
                end: bit_end,
            },
        }
    }
}

/// Extract a single bit from the provided `value`.
#[must_use]
pub fn extract_bit(value: u64, position: u32) -> u64 {
    extract_bits(value, position, position + 1)
}

/// Extract range-specified ([`start`..`end`]) bits from the provided `value`.
#[must_use]
pub fn extract_bits(value: u64, start: u32, end: u32) -> u64 {
    if start == 0 && end == u64::BITS {
        return value;
    }
    debug_assert!(start < end);
    (value >> (start)) & ((1 << (end - start)) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object::elf::*;

    #[test]
    fn test_rel_type_to_string() {
        assert_eq!(
            &x86_64_rel_type_to_string(R_X86_64_32),
            stringify!(R_X86_64_32)
        );
        assert_eq!(
            &x86_64_rel_type_to_string(R_X86_64_GOTPC32_TLSDESC),
            stringify!(R_X86_64_GOTPC32_TLSDESC)
        );
        assert_eq!(
            &x86_64_rel_type_to_string(64),
            "Unknown x86_64 relocation type 0x40"
        );

        assert_eq!(
            &aarch64_rel_type_to_string(64),
            "Unknown aarch64 relocation type 0x40"
        );
    }

    #[test]
    fn test_bit_operations() {
        assert_eq!(0b11000, extract_bits(0b1100_0000, 3, 8));
        assert_eq!(0b1010_1010_0000, extract_bits(0b10101010_00001111, 4, 16));
        assert_eq!(u32::MAX, extract_bits(u64::MAX, 0, 32) as u32);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_extract_bits_wrong_range() {
        let _ = extract_bits(0, 2, 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_extract_bits_too_large() {
        let _ = extract_bits(0, 0, 100);
    }
}
