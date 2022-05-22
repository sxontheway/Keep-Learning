#include <stdio.h>

extern "C" {
    void cfun(const double *indatav, size_t size, double *outdatav) 
    {
        size_t i;
        for (i = 0; i < size; ++i)
            outdatav[i] = indatav[i] * 2.0;
    }
}
