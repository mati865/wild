// Checked (non-_NC) PC-relative and TLS MOVW relocations must fit in the bits the instruction
// sequence can build: out-of-range values are an error, and in-range values link.
//#AbstractConfig:default
//#Arch:aarch64
//#Mode:static
//#RunEnabled:false
//#ReferenceLinkers:lld

//#Config:in-range:default

//#Config:prel-g0:default
//#CompArgs:-Wa,--defsym,PREL_G0=1
//#ExpectErrorWild:R_AARCH64_MOVW_PREL_G0
//#ExpectErrorWild:Relocation 65536 outside of bounds

//#Config:tprel-g0:default
//#CompArgs:-Wa,--defsym,TPREL_G0=1
//#ExpectErrorWild:R_AARCH64_TLSLE_MOVW_TPREL_G0
//#ExpectErrorWild:Relocation 65552 outside of bounds

.globl _start
_start:
.ifdef PREL_G0
  movn x0, #:prel_g0:.+0x10000
.endif
.ifdef TPREL_G0
  movz x0, #:tprel_g0:v1
.endif
  // The largest in-range value, which must keep linking.
  movn x0, #:prel_g0:.+0xffff
  ret

.section .tbss,"awT",@nobits
.space 0x10000
v1:
.space 8
