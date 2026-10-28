abs_outside = 0x5678;

ENTRY(_start)
PHDRS {
    headers PT_PHDR PHDRS;
    interp PT_INTERP;
    text PT_LOAD FILEHDR PHDRS FLAGS(5);
    data PT_LOAD FLAGS(6);
    dynamic PT_DYNAMIC;
}
SECTIONS {
    . = SIZEOF_HEADERS;
    .interp : { *(.interp) } :text :interp
    .dynsym : { *(.dynsym) } :text
    .dynstr : { *(.dynstr) } :text
    .hash : { *(.hash) } :text
    .gnu.hash : { *(.gnu.hash) } :text
    .rela.dyn : { *(.rela.dyn) } :text
    .relr.dyn : { *(.relr.dyn) } :text
    .text : {
        *(.text .text.*)
        absolute_value = ABSOLUTE(0x1233) + 1;
    } :text
    . = ALIGN(0x1000);
    .dynamic : { *(.dynamic) } :data :dynamic
    .data : { *(.data .data.*) } :data
}
