void crt_ctor();

__attribute__((used, section(".CRT$XCB"))) void (*const ctor_b0)(void) = crt_ctor;
