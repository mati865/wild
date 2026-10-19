//#Arch:aarch64
//#Compiler:clang
//#ReferenceLinkers:
//#RunEnabled:false
//#DiffEnabled:false
//#LinkArgs:-shared -z force-bti
//#ExpectDynamic:DT_AARCH64_BTI_PLT
//#ExpectWarning:.*: -z force-bti: file does not have BTI property
//#ExpectSectionBytes:.plt.got=0x5f2403d5 0..4
//#ExpectSectionBytes:.note.gnu.property=0x040000001000000005000000474e5500000000c0040000000100000000000000

.text
.globl exported_entry
.type exported_entry, %function
exported_entry:
    bl external_function
    ret
