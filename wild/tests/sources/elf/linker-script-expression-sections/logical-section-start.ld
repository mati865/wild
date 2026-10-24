ENTRY(_start)
SECTIONS {
    .text 0x10000 : {
        section_start = .;
        *(.text*)
    }
}

start_and = section_start && 1;
start_or = section_start || 0;
start_and_zero = section_start && 0;
start_or_one = section_start || 1;
start_and_expr = (section_start != 0) && (section_start == 0x10000);
start_or_expr = (section_start == 0) || (section_start == 0x10000);
start_not_expr = !(section_start == 0);
start_not_expr_false = !(section_start == 0x10000);
