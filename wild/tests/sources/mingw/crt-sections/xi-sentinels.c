void crt_ctor();

__attribute__((used, section(".CRT$XIA"))) void (*const ctor_i0)(void) = crt_ctor;

__attribute__((used, section(".CRT$XIZ"))) void (*const ctor_i1)(void) = crt_ctor;
