#pragma once
#ifndef GBKFIT_FFTW3_UTIL_HPP
#define GBKFIT_FFTW3_UTIL_HPP

namespace gbkfit {
namespace fftw3 {

int init_threads(void);

void cleanup_threads(void);

}
} // namespace gbkfit

#endif // GBKFIT_FFTW3_UTIL_HPP

