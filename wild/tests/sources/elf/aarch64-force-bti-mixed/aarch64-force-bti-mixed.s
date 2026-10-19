// Test BTI property merging with marked and unmarked AArch64 objects.

//#AbstractConfig:default
//#Arch:aarch64
//#Compiler:clang
//#ReferenceLinkers:
//#RunEnabled:false
//#DiffEnabled:false
//#LinkArgs:-shared
//#Object:marked.s

// Without force-bti, the unmarked input clears the BTI property.
//#Config:unforced:default
//#NoSection:.note.gnu.property
//#NoDynamic:DT_AARCH64_BTI_PLT
//#ExpectSectionBytes:.plt.got=0x1f2003d5 12..16

// With force-bti, warn about the unmarked input and enable BTI.
//#Config:forced:default
//#LinkArgs:-z force-bti
//#ExpectWarning:.*: -z force-bti: file does not have BTI property
//#ExpectSectionBytes:.plt.got=0x5f2403d5 0..4
//#ExpectSectionBytes:.note.gnu.property=0x040000001000000005000000474e5500000000c0040000000100000000000000
//#ExpectDynamic:DT_AARCH64_BTI_PLT

.text
.globl exported_entry
.type exported_entry, %function
exported_entry:
    bl external_function
    ret
