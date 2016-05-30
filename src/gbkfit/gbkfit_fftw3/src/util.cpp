
#include "gbkfit/fftw3/util.hpp"

#include <fftw3.h>

namespace gbkfit {
namespace fftw3 {

unsigned int ref_count = 0;

void init_threads(void)
{
    if (ref_count == 0) {
        fftwf_init_threads();
    }

    ref_count++;
}



} // namespace fftw3
} // namespace gbkfit
