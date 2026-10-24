ENTRY(_start)

PHDRS {
    text PT_LOAD FILEHDR PHDRS;
    data PT_LOAD;
}

SECTIONS {
    .debug_info 0 : {
        *(.debug_info)
    } :NONE
    .debug_str 0 : {
        *(.debug_str)
    } :NONE
    .text 0x400000 : {
        *(.text*)
    } :text
    .data 0x800000 : {
        *(.data*)
    } :data
    .debug_line 0 : {
        *(.debug_line)
    } :NONE
    .bss : {
        KEEP(*(.bss*))
    } :data
}

ASSERT(ADDR(.bss) > 0x800000, "bss is not after data")
