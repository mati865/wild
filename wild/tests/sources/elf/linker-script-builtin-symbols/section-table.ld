SECTIONS {
	. = 0x400000 + SIZEOF_HEADERS;
    .text : {
        *(.text)
    }
}
