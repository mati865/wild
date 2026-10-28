//#Arch:x86_64
//#Object:versions.c:-E -P
//#CompArgs:-fPIC
//#LinkArgs:-shared -soname versions.so
//#ReferenceLinkers:lld,bfd
//#RunEnabled:false
//#Contains:V2499

int exported(void) { return 42; }
