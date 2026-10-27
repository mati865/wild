# -z ibt and -z shstk mark the output as IBT- and SHSTK-compatible (GNU_PROPERTY_X86_FEATURE_1_IBT /
# _SHSTK) even though the input has no CET properties. -z ibtplt is accepted: Wild's x86-64 PLT
# entries always start with `endbr64`. lld doesn't support these options, so bfd is the reference.
//#AbstractConfig:default
//#Arch:x86_64
//#Mode:static
//#RunEnabled:false
//#CompArgs:-fcf-protection=none
//#ReferenceLinkers:bfd

//#Config:ibt:default
//#LinkArgs:-z ibt
//#ExpectSectionBytes:.note.gnu.property=0x020000c00400000001000000 16..28

//#Config:ibtplt:default
//#LinkArgs:-z ibtplt

# -z ibt suppresses the missing-IBT part of -z cet-report, so the first error is about SHSTK.
//#Config:ibt-cet-report:default
//#LinkArgs:-z cet-report=error -z ibt
//#ExpectErrorWild:.*: -z cet-report: file does not have GNU_PROPERTY_X86_FEATURE_1_SHSTK property

//#Config:shstk:default
//#LinkArgs:-z shstk
//#ExpectSectionBytes:.note.gnu.property=0x020000c00400000002000000 16..28

//#Config:ibt-shstk:default
//#LinkArgs:-z ibt -z shstk
//#ExpectSectionBytes:.note.gnu.property=0x020000c00400000003000000 16..28

# With both -z ibt and -z shstk, -z cet-report has nothing to report, so the link succeeds.
//#Config:ibt-shstk-cet-report:default
//#LinkArgs:-z cet-report=error -z ibt -z shstk
//#ExpectSectionBytes:.note.gnu.property=0x020000c00400000003000000 16..28

.globl _start
_start:
  ret
