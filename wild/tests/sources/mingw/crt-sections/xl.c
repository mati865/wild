void crt_ctor();

__attribute__((used, section(".CRT$XLC"))) void (*const ctor_l0)(void) = crt_ctor;
