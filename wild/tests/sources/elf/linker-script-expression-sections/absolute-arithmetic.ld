ENTRY(_start)
SECTIONS {
    .text 0x10000 : {
        section_start = .;
        *(.text*)
    }
}

/* These operations produce absolute values even with a section operand. */
sum = section_start + 0x1000;
difference = section_start - 0x1000;
product = section_start * 2;
quotient = section_start / 2;
shifted = section_start >> 1;
reverse_difference = 0x20000 - section_start;
