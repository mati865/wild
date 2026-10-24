ENTRY(_start)

PHDRS {
    text PT_LOAD FILEHDR PHDRS;
}

SECTIONS {
    .text 0x1000 : {
        *(.text)
    } :text
}
