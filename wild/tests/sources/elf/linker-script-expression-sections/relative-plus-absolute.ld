ENTRY(_start)
absolute_offset = 0x10;

SECTIONS {
    .text 0x10000 : {
        section_start = .;
        *(.text*)
    }
}

/* A named absolute symbol exercises Relative/Absolute, rather than
   Relative/Number. GNU ld makes both sums absolute outside a section. */
derived = section_start + absolute_offset;
derived_reverse = absolute_offset + section_start;
