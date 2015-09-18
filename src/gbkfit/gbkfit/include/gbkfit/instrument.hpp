#pragma once
#ifndef GBKFIT_INSTRUMENT_HPP
#define GBKFIT_INSTRUMENT_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"

namespace gbkfit {

class instrument
{

private:

    float m_step_x;
    float m_step_y;
    float m_step_z;

    point_spread_function* m_psf;
    line_spread_function* m_lsf;

public:

    instrument(float step_x, float step_y, float step_z, point_spread_function* psf, line_spread_function* lsf);

    ~instrument();

    float get_step_x(void) const;
    float get_step_y(void) const;
    float get_step_z(void) const;

    ndshape get_recommended_psf_size_spec(void) const;
    ndshape get_recommended_psf_size_spat(void) const;
    ndshape get_recommended_psf_size_cube(void) const;

    ndshape get_recommended_psf_size_spec(float step) const;
    ndshape get_recommended_psf_size_spat(float step_x, float step_y) const;
    ndshape get_recommended_psf_size_cube(float step_x, float step_y, float step_z) const;

    ndarray_host* create_psf_spec_data(void) const;
    ndarray_host* create_psf_spat_data(void) const;
    ndarray_host* create_psf_cube_data(void) const;

    ndarray_host* create_psf_spec_data(int size) const;
    ndarray_host* create_psf_spat_data(int size_x, int size_y) const;
    ndarray_host* create_psf_cube_data(int size_x, int size_y, int size_z) const;

    ndarray_host* create_psf_cube_data(float step_x, float step_y, float step_z) const;

    ndarray_host* create_psf_spec_data(int size, float step) const;
    ndarray_host* create_psf_spat_data(int size_x, int size_y, float step_x, float step_y) const;
    ndarray_host* create_psf_cube_data(int size_x, int size_y, int size_z, float step_x, float step_y, float step_z) const;

private:

    ndarray_host* create_psf_cube_data(const ndarray_host* spec_data, const ndarray_host* spat_data) const;

}; // class instrument

} // namespace gbkfit

#endif // GBKFIT_INSTRUMENT_HPP
