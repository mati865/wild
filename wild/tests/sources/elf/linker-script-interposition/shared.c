//#AugmentLinkerScript:interposition.ld

int default_func(void) { return 10; }

extern int my_func(void);
extern int default_only(void);

int call_func(void) { return my_func(); }

int call_default(void) { return default_only(); }
