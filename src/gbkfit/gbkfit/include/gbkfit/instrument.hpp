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

    Instrument(float step_x,
               float step_y,
               float step_z,
               PointSpreadFunction* psf,
               LineSpreadFunction* lsf);

    ~Instrument();

    float get_step_x(void) const;

    float get_step_y(void) const;

    float get_step_z(void) const;

    NDShape get_psf_size_spat(void) const;

    NDShape get_psf_size_spec(void) const;

    NDShape get_psf_size_cube(void) const;

    NDShape get_psf_size_spat(float step_x, float step_y) const;

    NDShape get_psf_size_spec(float step_z) const;

    NDShape get_psf_size_cube(float step_x, float step_y, float step_z) const;

    std::unique_ptr<NDArrayHost> create_psf_data_spat(void) const;

    std::unique_ptr<NDArrayHost> create_psf_data_spec(void) const;

    std::unique_ptr<NDArrayHost> create_psf_data_cube(void) const;

    std::unique_ptr<NDArrayHost> create_psf_data_spat(float step_x,
                                                      float step_y) const;

    std::unique_ptr<NDArrayHost> create_psf_data_spec(float step_z) const;

    std::unique_ptr<NDArrayHost> create_psf_data_cube(float step_x,
                                                      float step_y,
                                                      float step_z) const;

    std::unique_ptr<NDArrayHost> create_psf_data_spat(float step_x,
                                                      float step_y,
                                                      int size_x,
                                                      int size_y) const;

    std::unique_ptr<NDArrayHost> create_psf_data_spec(float step_z,
                                                      int size_z) const;

    std::unique_ptr<NDArrayHost> create_psf_data_cube(float step_x,
                                                      float step_y,
                                                      float step_z,
                                                      int size_x,
                                                      int size_y,
                                                      int size_z) const;

public:

    std::unique_ptr<NDArrayHost> create_psf_data_cube(
            const std::unique_ptr<NDArrayHost>& data_spat,
            const std::unique_ptr<NDArrayHost>& data_spec) const;

};

} // namespace gbkfit

#endif // GBKFIT_INSTRUMENT_HPP
