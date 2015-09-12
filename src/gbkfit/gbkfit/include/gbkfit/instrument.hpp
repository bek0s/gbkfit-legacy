#pragma once
#ifndef GBKFIT_INSTRUMENT_HPP
#define GBKFIT_INSTRUMENT_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

class instrument
{

private:

    int m_sampling_x;
    int m_sampling_y;
    int m_sampling_z;

    point_spread_function* m_psf;
    line_spread_function* m_lsf;

    ndarray_host* m_psf_spat;
    ndarray_host* m_psf_spec;
    ndarray_host* m_psf_cube;

public:

    instrument(int sampling_x, int sampling_y, int sampling_z, point_spread_function* psf, line_spread_function* lsf);

    ~instrument();

    int get_step_x(void) const;

    int get_step_y(void) const;

    int get_step_z(void) const;

    point_spread_function* get_psf(void) { return m_psf; }

    line_spread_function* get_lsf(void) { return m_lsf; }

    ndarray_host* get_psf_cube(void) { return m_psf_cube; }


    ndarray_host* create_psf_spec_data(void) const;
    ndarray_host* create_psf_spat_data(void) const;
    ndarray_host* create_psf_cube_data(void) const;

    ndarray_host* create_psf_spec_data(int size) const;
    ndarray_host* create_psf_spat_data(int size_x, int size_y) const;
    ndarray_host* create_psf_cube_data(int size_x, int size_y, int size_z) const;

    ndarray_host* create_psf_spec_data(int size, float step) const;
    ndarray_host* create_psf_spat_data(int size_x, int size_y, float step_x, float step_y) const;
    ndarray_host* create_psf_cube_data(int size_x, int size_y, int size_z, float step_x, float step_y, float step_z) const;




    std::unique_ptr<ndarray_host> get_psf_spec_data(void) const;
    std::unique_ptr<ndarray_host> get_psf_spat_data(void) const;
    std::unique_ptr<ndarray_host> get_psf_cube_data(void) const;

    std::unique_ptr<ndarray_host> get_psf_spec_data(int size_z) const;
    std::unique_ptr<ndarray_host> get_psf_spat_data(int size_x, int size_y) const;
    std::unique_ptr<ndarray_host> get_psf_cube_data(int size_x, int size_y, int size_z) const;

}; // class instrument

} // namespace gbkfit

#endif // GBKFIT_INSTRUMENT_HPP
