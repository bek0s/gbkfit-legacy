#pragma once
#ifndef GBKFIT_INSTRUMENT_HPP
#define GBKFIT_INSTRUMENT_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"

namespace gbkfit {

class Instrument
{

private:

    float m_step_x;
    float m_step_y;
    float m_step_z;

    PointSpreadFunction* m_psf;
    LineSpreadFunction* m_lsf;

public:

    Instrument(float step_x, float step_y, float step_z, PointSpreadFunction* psf, LineSpreadFunction* lsf);

    ~Instrument();

    float get_step_x(void) const;
    float get_step_y(void) const;
    float get_step_z(void) const;

    NDShape get_recommended_psf_size_spec(void) const;
    NDShape get_recommended_psf_size_spat(void) const;
    NDShape get_recommended_psf_size_cube(void) const;

    NDShape get_recommended_psf_size_spec(float step) const;
    NDShape get_recommended_psf_size_spat(float step_x, float step_y) const;
    NDShape get_recommended_psf_size_cube(float step_x, float step_y, float step_z) const;

    NDArrayHost* create_psf_spec_data(void) const;
    NDArrayHost* create_psf_spat_data(void) const;
    NDArrayHost* create_psf_cube_data(void) const;

    NDArrayHost* create_psf_spec_data(int size) const;
    NDArrayHost* create_psf_spat_data(int size_x, int size_y) const;
    NDArrayHost* create_psf_cube_data(int size_x, int size_y, int size_z) const;

    NDArrayHost* create_psf_cube_data(float step_x, float step_y, float step_z) const;

    NDArrayHost* create_psf_spec_data(int size, float step) const;
    NDArrayHost* create_psf_spat_data(int size_x, int size_y, float step_x, float step_y) const;
    NDArrayHost* create_psf_cube_data(int size_x, int size_y, int size_z, float step_x, float step_y, float step_z) const;

private:

    NDArrayHost* create_psf_cube_data(const NDArrayHost* spec_data, const NDArrayHost* spat_data) const;

}; // class instrument

} // namespace gbkfit

#endif // GBKFIT_INSTRUMENT_HPP
