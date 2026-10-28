abs_outside = 0x5678;

ENTRY(_start)
PHDRS {
    headers PT_PHDR PHDRS;
    text PT_LOAD FILEHDR PHDRS FLAGS(5);
    data PT_LOAD FLAGS(6);
}
SECTIONS {
    . = 0x400000 + SIZEOF_HEADERS;
    .text : {
        *(.text .text.*)
        absolute_value = ABSOLUTE(0x1233) + 1;
    } :text
    . = ALIGN(0x1000);
    .data : { *(.data .data.*) } :data
}
