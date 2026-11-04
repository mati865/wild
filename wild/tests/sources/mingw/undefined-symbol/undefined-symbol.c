//#ExpectErrorWild: Undefined symbol missing_function

extern int missing_function(void);

int mainCRTStartup(void) {
    return missing_function();
}
