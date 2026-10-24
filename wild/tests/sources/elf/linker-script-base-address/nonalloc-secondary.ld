ENTRY(_start)

PHDRS {
    text PT_LOAD FILEHDR PHDRS;
    data PT_LOAD;
}

SECTIONS {
    .metadata 0 : {
        KEEP(*(.meta.first))
        KEEP(*(.meta.second))
    } :NONE
    .text 0x400000 : {
        *(.text*)
    } :text
    .data 0x800000 : {
        *(.data*)
    } :data
    .bss : {
        *(.bss*)
    } :data
}
