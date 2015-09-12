
#include "gbkfit/instrument.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/spread_function.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/utility.hpp"
#include "gbkfit/math.hpp"

namespace gbkfit
{

instrument::instrument(int sampling_x, int sampling_y, int sampling_z, point_spread_function* psf, line_spread_function* lsf)
    : m_sampling_x(sampling_x)
    , m_sampling_y(sampling_y)
    , m_sampling_z(sampling_z)
    , m_psf(psf)
    , m_lsf(lsf)
{

    /*
    int psf_size_x = util_num::roundu_odd((10 * 2.0) / m_sampling_x);
    int psf_size_y = util_num::roundu_odd((10 * 2.0) / m_sampling_y);
    int psf_size_z = util_num::roundu_odd((10 * 30.0) / m_sampling_z);

    m_psf_spat = new ndarray_host_new({psf_size_x, psf_size_y});
    m_psf_spec = new ndarray_host_new({psf_size_z});
    m_psf_cube = new ndarray_host_new({psf_size_x, psf_size_y, psf_size_z});

    m_psf->as_image(psf_size_x,psf_size_y,m_sampling_x,m_sampling_y,m_psf_spat->get_host_ptr());
    m_lsf->as_array(psf_size_z,m_sampling_z,m_psf_spec->get_host_ptr());

    float* psf_spat_raw = m_psf_spat->get_host_ptr();
    float* psf_spec_raw = m_psf_spec->get_host_ptr();
    float* psf_cube_raw = m_psf_cube->get_host_ptr();

    for(int z = 0; z < psf_size_z; ++z)
    {
        for(int y = 0; y < psf_size_y; ++y)
        {
            for(int x = 0; x < psf_size_x; ++x)
            {
                int idx_3d = z*psf_size_x*psf_size_y + y*psf_size_x + x;
                int idx_2d = y*psf_size_x + x;
                int idx_1d = z;

                psf_cube_raw[idx_3d] = psf_spat_raw[idx_2d] * psf_spec_raw[idx_1d];
            }
        }
    }

    math::normalize_integral(psf_cube_raw,m_psf_cube->get_shape().get_dim_length_product());

//  std::cout << std::accumulate(psf_cube_raw,psf_cube_raw+m_psf_cube->get_shape().get_dim_length_product(),static_cast<float>(0)) << std::endl;

    fits::write_to("!psf_spat.fits",*m_psf_spat);
    fits::write_to("!psf_spec.fits",*m_psf_spec);
    fits::write_to("!psf_cube.fits",*m_psf_cube);
    */


}

instrument::~instrument()
{
    delete m_psf;
    delete m_lsf;

    delete m_psf_spat;
    delete m_psf_spec;
    delete m_psf_cube;
}

int instrument::get_step_x(void) const
{
    return m_sampling_x;
}

int instrument::get_step_y(void) const
{
    return m_sampling_y;
}

int instrument::get_step_z(void) const
{
    return m_sampling_z;
}

std::unique_ptr<ndarray_host> instrument::get_psf_spec_data(void) const
{
    int size_z = 27;
    return get_psf_spec_data(size_z);
}

std::unique_ptr<ndarray_host> instrument::get_psf_spat_data(void) const
{
    int size_x = 17;
    int size_y = 17;
    return get_psf_spat_data(size_x,size_y);
}

std::unique_ptr<ndarray_host> instrument::get_psf_cube_data(void) const
{
    int size_x = 17; // m_psf->get_recommended_size_x(m_sampling_x);
    int size_y = 17; // m_psf->get_recommended_size_y(m_sampling_y);
    int size_z = 27; // m_lsf->get_recommended_size_z(m_sampling_z);
    return get_psf_cube_data(size_x,size_y,size_z);
}

std::unique_ptr<ndarray_host> instrument::get_psf_spec_data(int size_z) const
{
}

std::unique_ptr<ndarray_host> instrument::get_psf_spat_data(int size_x, int size_y) const
{
}

std::unique_ptr<ndarray_host> instrument::get_psf_cube_data(int size_x, int size_y, int size_z) const
{
}





} // namespace gbkfit
