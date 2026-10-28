SECTIONS {
  . = 0x1000;
  .data : { *(.data) }
  .odd 0x2001 : { *(.odd) }
}
