
#include <cuda_runtime_api.h>

#include <cufft.h>

#include "gbkfit/dmodel/scube/scube_cuda.hpp"
#include "gbkfit/dmodel/scube/scube_cuda_factory.hpp"
#include "gbkfit/dmodel/scube/scube_cuda_kernels_h.hpp"

#include "gbkfit/cuda/ndarray.hpp"

#include "gbkfit/array_util.hpp"
#include "gbkfit/gmodel.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/spread_functions.hpp"

#include "gbkfit/utility.hpp"

namespace gbkfit {
namespace dmodel {
namespace scube {

SCubeCuda::SCubeCuda(int size_x,
                     int size_y,
                     int size_z,
                     float step_x,
                     float step_y,
                     float step_z,
                     int upsampling_x,
                     int upsampling_y,
                     int upsampling_z,
                     const PointSpreadFunction* psf,
                     const LineSpreadFunction* lsf)
    :
      m_h_flxcube(nullptr)
    , m_h_flxcube_up(nullptr)
    , m_h_psfcube(nullptr)
    , m_h_psfcube_up(nullptr)
    , m_d_flxcube(nullptr)
    , m_d_flxcube_up(nullptr)
    , m_d_flxcube_up_fft(nullptr)
    , m_d_psfcube(nullptr)
    , m_d_psfcube_u(nullptr)
    , m_d_psfcube_up(nullptr)
    , m_d_psfcube_up_fft(nullptr)
{
    m_upsampling = {
        upsampling_x,
        upsampling_y,
        upsampling_z
    };

    //
    // Calculate original sizes
    //

    m_flxcube_size = {
        size_x,
        size_y,
        size_z
    };

    m_step = {
        step_x,
        step_y,
        step_z
    };

    m_psfcube_size = spread_function_util::get_psf_cube_size(
            psf,
            lsf,
            m_step[0],
            m_step[1],
            m_step[2]).as_vector();

    //
    // Calculate upsampled sizes
    //

    m_flxcube_size_u = {
        m_flxcube_size[0]*m_upsampling[0],
        m_flxcube_size[1]*m_upsampling[1],
        m_flxcube_size[2]*m_upsampling[2]
    };

    m_step_u = {
        m_step[0]/m_upsampling[0],
        m_step[1]/m_upsampling[1],
        m_step[2]/m_upsampling[2]
    };

    m_psfcube_size_u = spread_function_util::get_psf_cube_size(
            psf,
            lsf,
            m_step_u[0],
            m_step_u[1],
            m_step_u[2]).as_vector();

    //
    // Calculate upsampled+padded sizes.
    // Padding includes (upsampled) PSF and power-of-two padding  padding.
    //

    m_size_up = {
        m_flxcube_size_u[0] + m_psfcube_size_u[0] - 1,
        m_flxcube_size_u[1] + m_psfcube_size_u[1] - 1,
        m_flxcube_size_u[2] + m_psfcube_size_u[2] - 1
    };

    m_size_up[0] = util_fft::calculate_optimal_dim_length(static_cast<std::uint32_t>(m_size_up[0]), 512, 256);
    m_size_up[1] = util_fft::calculate_optimal_dim_length(static_cast<std::uint32_t>(m_size_up[1]), 512, 256);
    m_size_up[2] = util_fft::calculate_optimal_dim_length(static_cast<std::uint32_t>(m_size_up[2]), 512, 256);

    //
    // Pixel count (original, upsampled, fft)
    //

    int flxcube_size_1d = m_flxcube_size[2]*m_flxcube_size[1]*m_flxcube_size[0];
    int size_1d_up = m_size_up[2]*m_size_up[1]*m_size_up[0];
    int size_1d_up_fft = m_size_up[2]*m_size_up[1]*(m_size_up[0]/2+1);

    //
    // Allocate and initialize flux cubes
    //

    m_h_flxcube = new NDArrayHost(m_flxcube_size);
    m_h_flxcube_up = new NDArrayHost(m_size_up);

    std::fill_n(m_h_flxcube->get_host_ptr(), flxcube_size_1d, -1);
    std::fill_n(m_h_flxcube_up->get_host_ptr(), size_1d_up, -1);

    m_d_flxcube = new cuda::NDArrayManaged(m_flxcube_size);
    m_d_flxcube_up = new cuda::NDArrayManaged(m_size_up);
    m_d_flxcube_up_fft = new cuda::NDArrayManaged({2*size_1d_up_fft});

    //std::cout <<"OMG"<< std::endl;
    std::fill_n(m_d_flxcube->get_cuda_ptr(), flxcube_size_1d, -1);
    std::fill_n(m_d_flxcube_up->get_cuda_ptr(), size_1d_up, -1);
    std::fill_n(m_d_flxcube_up_fft->get_cuda_ptr(), 2*size_1d_up_fft, -1);

    cudaDeviceSynchronize();

    //
    // Allocate and initialize psf cubes
    //

    m_h_psfcube = spread_function_util::create_psf_cube(
            psf,
            lsf,
            m_step[0],
            m_step[1],
            m_step[2]).release();

    m_h_psfcube_u = spread_function_util::create_psf_cube(
            psf,
            lsf,
            m_step_u[0],
            m_step_u[1],
            m_step_u[2]).release();

    m_h_psfcube_up = new NDArrayHost(m_size_up);

    array_util::array_copy(m_psfcube_size_u[0],
                           m_psfcube_size_u[1],
                           m_psfcube_size_u[2],
                           m_size_up[0],
                           m_size_up[1],
                           m_size_up[2],
                           m_h_psfcube_u->get_host_ptr(),
                           m_h_psfcube_up->get_host_ptr());

    array_util::array_fill(0,
                           m_psfcube_size_u[0],
                           m_psfcube_size_u[1],
                           m_psfcube_size_u[2],
                           m_size_up[0],
                           m_size_up[1],
                           m_size_up[2],
                           m_h_psfcube_up->get_host_ptr());

    array_util::array_shift(m_size_up[0],
                            m_size_up[1],
                            m_size_up[2],
                            -m_psfcube_size_u[0]/2,
                            -m_psfcube_size_u[1]/2,
                            -m_psfcube_size_u[2]/2,
                            m_h_psfcube_up->get_host_ptr());


    m_d_psfcube = new cuda::NDArrayManaged(m_psfcube_size);
    m_d_psfcube_up = new cuda::NDArrayManaged(m_size_up);
    m_d_psfcube_up_fft = new cuda::NDArrayManaged({2*size_1d_up_fft});



    m_d_psfcube->write_data(m_h_psfcube->get_host_ptr());
    m_d_psfcube_up->write_data(m_h_psfcube_up->get_host_ptr());

    cudaDeviceSynchronize();

    //
    // Create fft plans for flux and psf cubes
    //

    cufftPlan3d(&m_fft_plan_flxcube_r2c,
                m_size_up[2],
                m_size_up[1],
                m_size_up[0],
                CUFFT_R2C);

    cufftPlan3d(&m_fft_plan_flxcube_c2r,
                m_size_up[2],
                m_size_up[1],
                m_size_up[0],
                CUFFT_C2R);

    cufftPlan3d(&m_fft_plan_psfcube_r2c,
                m_size_up[2],
                m_size_up[1],
                m_size_up[0],
                CUFFT_R2C);

    cudaDeviceSynchronize();

    //
    // FFT-transform the PSF cube
    //

    cufftExecR2C(m_fft_plan_psfcube_r2c,
                 m_d_psfcube_up->get_cuda_ptr(),
                 (cufftComplex*)m_d_psfcube_up_fft->get_cuda_ptr());

    cudaDeviceSynchronize();

    //
    // Add output data to the output data map
    //

//  m_h_output_map["psfcube"] = m_h_psfcube;
    m_h_output_map["flxcube"] = m_h_flxcube;
//  m_h_output_map["flxcube_up"] = m_h_flxcube_up;

//  m_d_output_map["psfcube"] = m_d_psfcube;
    m_d_output_map["flxcube"] = m_d_flxcube;
//  m_d_output_map["flxcube_up"] = m_d_flxcube_up;
}

SCubeCuda::~SCubeCuda()
{
    delete m_h_flxcube;
    delete m_h_flxcube_up;

    delete m_h_psfcube;
    delete m_h_psfcube_u;
    delete m_h_psfcube_up;

    delete m_d_flxcube;
    delete m_d_flxcube_up;
    delete m_d_flxcube_up_fft;

    delete m_d_psfcube;
    delete m_d_psfcube_up;
    delete m_d_psfcube_up_fft;

    cufftDestroy(m_fft_plan_flxcube_r2c);
    cufftDestroy(m_fft_plan_flxcube_c2r);
    cufftDestroy(m_fft_plan_psfcube_r2c);
}

const std::string& SCubeCuda::get_type(void) const
{
    return SCubeCudaFactory::FACTORY_TYPE;
}

const std::vector<int>& SCubeCuda::get_size(void) const
{
    return m_flxcube_size;
}

const std::vector<float>& SCubeCuda::get_step(void) const
{
    return m_step;
}

const std::map<std::string, NDArrayHost*>& SCubeCuda::evaluate(
        const std::map<std::string, float>& params) const
{
    evaluate_managed(params);

    m_d_flxcube->read_data(m_h_flxcube->get_host_ptr());
    m_d_flxcube_up->read_data(m_h_flxcube_up->get_host_ptr());

    return m_h_output_map;
}

const std::map<std::string, cuda::NDArrayManaged*>& SCubeCuda::evaluate_managed(
        const std::map<std::string, float>& params) const
{
    // Define the coordinates (in pixels) of the zero point of the cube
    std::vector<float> zero = {
        -m_step_u[0]*(m_psfcube_size_u[0]/2),
        -m_step_u[1]*(m_psfcube_size_u[1]/2),
        -m_step_u[2]*(m_psfcube_size_u[2]/2 + m_flxcube_size_u[2]/2)
    };

    // Evaluate model on the cube
    m_gmodel->evaluate(params, zero, m_step_u, m_d_flxcube_up);

    // Perform convolution with the psf
    kernels_cuda_h::convolve_cube(m_d_flxcube_up->get_cuda_ptr(),
            reinterpret_cast<cufftComplex*>(m_d_flxcube_up_fft->get_cuda_ptr()),
            reinterpret_cast<cufftComplex*>(m_d_psfcube_up_fft->get_cuda_ptr()),
            m_size_up[0],
            m_size_up[1],
            m_size_up[2],
            m_fft_plan_flxcube_r2c,
            m_fft_plan_flxcube_c2r);

    // Downsample the upsampled cube to the original size
    int offset_x = m_psfcube_size_u[0]/2;
    int offset_y = m_psfcube_size_u[1]/2;
    int offset_z = m_psfcube_size_u[2]/2;
    kernels_cuda_h::downsample_cube(m_flxcube_size[0],
                                    m_flxcube_size[1],
                                    m_flxcube_size[2],
                                    m_size_up[0],
                                    m_size_up[1],
                                    m_size_up[2],
                                    offset_x,
                                    offset_y,
                                    offset_z,
                                    m_upsampling[0],
                                    m_upsampling[1],
                                    m_upsampling[2],
                                    m_d_flxcube_up->get_cuda_ptr(),
                                    m_d_flxcube->get_cuda_ptr());

    return m_d_output_map;
}

} // namespace scube
} // namespace dmodel
} // namespace gbkfit
