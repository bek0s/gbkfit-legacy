
#include "gbkfit/dmodel/scube/scube_omp.hpp"

#include <omp.h>

#include "gbkfit/dmodel/scube/scube_omp_factory.hpp"
#include "gbkfit/dmodel/scube/scube_omp_kernels.hpp"

#include "gbkfit/array_util.hpp"
#include "gbkfit/gmodel.hpp"
#include "gbkfit/instrument.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/utility.hpp"

namespace gbkfit {
namespace dmodel {
namespace scube {

SCubeOmp::SCubeOmp(int size_x,
                   int size_y,
                   int size_z,
                   const Instrument* instrument)
    : SCubeOmp(size_x, size_y, size_z, 1, 1, 1, instrument)
{
}

SCubeOmp::SCubeOmp(int size_x,
                   int size_y,
                   int size_z,
                   int upsampling_x,
                   int upsampling_y,
                   int upsampling_z,
                   const Instrument* instrument)
    : m_instrument(instrument)
{
    m_size = {
        size_x,
        size_y,
        size_z
    };

    m_upsampling = {
        upsampling_x,
        upsampling_y,
        upsampling_z
    };

    m_step = {
        instrument->get_step_x(),
        instrument->get_step_y(),
        instrument->get_step_z()
    };

    m_psfcube_size = instrument->get_psf_size_cube(m_step[0],
                                                   m_step[1],
                                                   m_step[2]).as_vector();

    //
    // Calculate upsampled sizes
    //

    m_size_u = {
        m_size[0]*m_upsampling[0],
        m_size[1]*m_upsampling[1],
        m_size[2]*m_upsampling[2]
    };

    m_step_u = {
        m_step[0]/m_upsampling[0],
        m_step[1]/m_upsampling[1],
        m_step[2]/m_upsampling[2]
    };

    m_psfcube_size_u = instrument->get_psf_size_cube(m_step_u[0],
                                                 m_step_u[1],
                                                 m_step_u[2]).as_vector();

    //
    // Calculate final size for both data and psf cubes
    //

    m_size_up = {
        m_size_u[0] + m_psfcube_size_u[0] - 1,
        m_size_u[1] + m_psfcube_size_u[1] - 1,
        m_size_u[2] + m_psfcube_size_u[2] - 1
    };


    m_size_up[0] = util_fft::calculate_optimal_dim_length(static_cast<std::uint32_t>(m_size_up[0]), 512, 256);
    m_size_up[1] = util_fft::calculate_optimal_dim_length(static_cast<std::uint32_t>(m_size_up[1]), 512, 256);
    m_size_up[2] = util_fft::calculate_optimal_dim_length(static_cast<std::uint32_t>(m_size_up[2]), 512, 256);

    int size_1d = m_size[2]*m_size[1]*m_size[0];
    int size_1d_up = m_size_up[2]*m_size_up[1]*m_size_up[0];
    int size_1d_up_fft = m_size_up[2]*m_size_up[1]*(m_size_up[0]/2+1);

    //
    // Allocate and initialize flux cubes
    //

    m_flxcube = new NDArrayHost(m_size);
    m_flxcube_up = new NDArrayHost(m_size_up);
    m_flxcube_up_fft = new NDArrayHost({2*size_1d_up_fft});

    std::fill_n(m_flxcube->get_host_ptr(), size_1d, -1);

    std::fill_n(m_flxcube_up->get_host_ptr(), size_1d_up, -1);

    std::fill_n(m_flxcube_up_fft->get_host_ptr(), 2*size_1d_up_fft, -1);

    //
    // Allocate and initialize psf cubes
    //

    m_psfcube = instrument->create_psf_data_cube(m_step[0],
                                                 m_step[1],
                                                 m_step[2]).release();
    m_psfcube_up = instrument->create_psf_data_cube(m_step_u[0],
                                                    m_step_u[1],
                                                    m_step_u[2],
                                                    m_size_up[0],
                                                    m_size_up[1],
                                                    m_size_up[2]).release();
    m_psfcube_up_fft = new NDArrayHost({2*size_1d_up_fft});

    array_util::array_shift(m_size_up[0],
                            m_size_up[1],
                            m_size_up[2],
                            -m_size_up[0]/2,
                            -m_size_up[1]/2,
                            -m_size_up[2]/2,
                            m_psfcube_up->get_host_ptr());

    std::fill_n(m_psfcube_up_fft->get_host_ptr(), 2*size_1d_up_fft, -1);

    //
    // Create fft plans for flux and psf cubes.
    //

    fftwf_init_threads();

    fftwf_plan_with_nthreads(omp_get_max_threads());

    m_fft_plan_flxcube_r2c = fftwf_plan_dft_r2c_3d(
            m_size_up[2],
            m_size_up[1],
            m_size_up[0],
            m_flxcube_up->get_host_ptr(),
            (fftwf_complex*)m_flxcube_up_fft->get_host_ptr(),
            FFTW_ESTIMATE);

    m_fft_plan_flxcube_c2r = fftwf_plan_dft_c2r_3d(
            m_size_up[2],
            m_size_up[1],
            m_size_up[0],
            (fftwf_complex*)m_flxcube_up_fft->get_host_ptr(),
            m_flxcube_up->get_host_ptr(),
            FFTW_ESTIMATE);

    m_fft_plan_psfcube_r2c = fftwf_plan_dft_r2c_3d(
            m_size_up[2],
            m_size_up[1],
            m_size_up[0],
            m_psfcube_up->get_host_ptr(),
            (fftwf_complex*)m_psfcube_up_fft->get_host_ptr(),
            FFTW_ESTIMATE);

    //
    // FFT-transform the PSF cube
    //

    fftwf_execute_dft_r2c(
            m_fft_plan_psfcube_r2c,
            m_psfcube_up->get_host_ptr(),
            (fftwf_complex*)m_psfcube_up_fft->get_host_ptr());

    //
    // Add output data to the output data map
    //

    m_output_map["flxcube"] = m_flxcube;
    m_output_map["psfcube"] = m_psfcube;

    m_output_map["flxcube_up"] = m_flxcube_up;
    m_output_map["psfcube_up"] = m_psfcube_up;
}

SCubeOmp::~SCubeOmp()
{


    fftwf_destroy_plan(m_fft_plan_flxcube_r2c);
    fftwf_destroy_plan(m_fft_plan_flxcube_c2r);
    fftwf_destroy_plan(m_fft_plan_psfcube_r2c);

//  fftwf_cleanup();
    //fftwf_cleanup_threads();

    delete m_flxcube;
    delete m_flxcube_up;
    delete m_flxcube_up_fft;

    delete m_psfcube;
    delete m_psfcube_up;
    delete m_psfcube_up_fft;


}

const std::string& SCubeOmp::get_type(void) const
{
    return SCubeOmpFactory::FACTORY_TYPE;
}

const Instrument* SCubeOmp::get_instrument(void) const
{
    return m_instrument;
}

const std::map<std::string, NDArrayHost*>& SCubeOmp::evaluate(
        const std::map<std::string, float>& params) const
{
    //
    // Define the coordinates of the zero point of the cube.
    //
    // Spatial coordinates: The values correspond to the bottom left corner of
    // the first pixel.
    //
    // Spectral coordinate: The value corresponds to the middle of the first
    // pixel.
    //
    std::vector<float> zero = {
        -(m_psfcube_size_u[0]/2)*m_step_u[0],
        -(m_psfcube_size_u[1]/2)*m_step_u[1],

        0 - (m_psfcube_size_u[2]/2 + m_size_u[2]/2)*m_step_u[2]
    };


    m_gmodel->evaluate(params, zero, m_step_u, m_flxcube_up);

    int offset_x = m_psfcube_size_u[0]/2;
    int offset_y = m_psfcube_size_u[1]/2;
    int offset_z = m_psfcube_size_u[2]/2;

    /*
    kernels_omp::model_image_3d_convolve_fft(m_flxcube_up->get_host_ptr(),
                                             reinterpret_cast<fftwf_complex*>(m_flxcube_up_fft->get_host_ptr()),
                                             reinterpret_cast<fftwf_complex*>(m_psfcube_up_fft->get_host_ptr()),
                                             m_size_up[0],
                                             m_size_up[1],
                                             m_size_up[2],
                                             m_fft_plan_flxcube_r2c,
                                             m_fft_plan_flxcube_c2r);*/

    kernels_omp::downsample_cube(m_size[0],
                                 m_size[1],
                                 m_size[2],
                                 m_size_up[0],
                                 m_size_up[1],
                                 m_size_up[2],
                                 offset_x,
                                 offset_y,
                                 offset_z,
                                 m_upsampling[0],
                                 m_upsampling[1],
                                 m_upsampling[2],
                                 m_flxcube_up->get_host_ptr(),
                                 m_flxcube->get_host_ptr());

    return m_output_map;
}

} // namespace scube
} // namespace dmodel
} // namespace gbkfit
