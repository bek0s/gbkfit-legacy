
#include "gbkfit/model_thindisk/model_thindisk.hpp"
#include "gbkfit/model_thindisk/model_thindisk_kernels_omp.hpp"
#include "gbkfit/num_util.hpp"
#include "gbkfit/image_util.hpp"
#include <iostream>

namespace gbkfit {
namespace model_thindisk {

std::size_t calculate_fft_dim_length(std::size_t length)
{
    std::size_t new_length = gbkfit::num_util::roundup_po2(length);

    if(new_length > 512)
    {
        new_length = gbkfit::num_util::roundup_multiple(new_length,(std::size_t)256);
    }

    return new_length;
}

model_thindisk::model_thindisk(std::size_t width,
                               std::size_t height,
                               std::size_t depth,
                               float step_x,
                               float step_y,
                               float step_z,
                               std::size_t upsampling_x,
                               std::size_t upsampling_y,
                               std::size_t upsampling_z,
                               const gbkfit::ndarray* psf)
    : m_intcube_width(width),
      m_intcube_height(height),
      m_intcube_depth(depth),
      m_step_x(step_x),
      m_step_y(step_y),
      m_step_z(step_z),
      m_upsampling_x(upsampling_x),
      m_upsampling_y(upsampling_y),
      m_upsampling_z(upsampling_z)
{
    m_intcube_aligned_width = m_intcube_width * m_upsampling_x;
    m_intcube_aligned_height = m_intcube_height * m_upsampling_y;
    m_intcube_aligned_depth = m_intcube_depth * m_upsampling_z;

    m_psf_width = 0;
    m_psf_height = 0;
    m_psf_depth = 0;

    m_h_intcube = nullptr;
    m_h_intcube_aligned = nullptr;
    m_h_intcube_aligned_fftw3 = nullptr;

    m_h_psf = nullptr;
    m_h_psf_aligned = nullptr;
    m_h_psf_aligned_fftw3 = nullptr;

    m_h_velmap = nullptr;
    m_h_sigmap = nullptr;

    m_h_velmap_aligned = nullptr;
    m_h_sigmap_aligned = nullptr;

    m_fft_plan_psf_r2c = nullptr;
    m_fft_plan_intcube_r2c = nullptr;
    m_fft_plan_intcube_c2r = nullptr;

    /*
    omp_set_num_threads(64);
    fftwf_init_threads();
    fftwf_plan_with_nthreads(64);
    */

    if(psf)
    {
        m_psf_width = psf->get_shape().get_dim_length(0);
        m_psf_height = psf->get_shape().get_dim_length(1);
        m_psf_depth = psf->get_shape().get_dim_length(2);

        m_intcube_aligned_width = calculate_fft_dim_length(m_intcube_aligned_width+m_psf_width-1);
        m_intcube_aligned_height = calculate_fft_dim_length(m_intcube_aligned_height+m_psf_height-1);
        m_intcube_aligned_depth = calculate_fft_dim_length(m_intcube_aligned_depth+m_psf_depth-1);

        std::size_t psf_length = m_psf_depth*m_psf_height*m_psf_width;
        std::size_t psf_aligned_length = m_intcube_aligned_depth*m_intcube_aligned_height*m_intcube_aligned_width;
        std::size_t psf_aligned_fft_length = m_intcube_aligned_depth*m_intcube_aligned_height*(m_intcube_aligned_width/2+1);

        m_h_psf = reinterpret_cast<float*>(fftwf_malloc(psf_length*sizeof(float)));
        m_h_psf_aligned = reinterpret_cast<float*>(fftwf_malloc(psf_aligned_length*sizeof(float)));
        m_h_psf_aligned_fftw3 = reinterpret_cast<std::complex<float>*>(fftwf_malloc(psf_aligned_fft_length*sizeof(std::complex<float>)));

        m_fft_plan_psf_r2c = fftwf_plan_dft_r2c_3d(m_intcube_aligned_depth,m_intcube_aligned_height,m_intcube_aligned_width,m_h_psf_aligned,reinterpret_cast<fftwf_complex*>(m_h_psf_aligned_fftw3),FFTW_ESTIMATE);

    //  std::copy(std::begin(psf->get_data()),std::end(psf->get_data()),m_h_psf);

        psf->read_data(m_h_psf);

        int margin_width_0, margin_width_1, margin_height_0, margin_height_1, margin_depth_0, margin_depth_1;
        get_intcube_margins(margin_width_0,margin_width_1,margin_height_0,margin_height_1,margin_depth_0,margin_depth_1);

        image_util::img_pad_copy(m_h_psf_aligned,m_h_psf,m_intcube_aligned_width,m_intcube_aligned_height,m_intcube_aligned_depth,m_psf_width,m_psf_height,m_psf_depth);
        image_util::img_pad_fill_value(m_h_psf_aligned,m_intcube_aligned_width,m_intcube_aligned_height,m_intcube_aligned_depth,m_psf_width,m_psf_height,m_psf_depth,0);
        image_util::img_circular_shift(m_h_psf_aligned,m_intcube_aligned_width,m_intcube_aligned_height,m_intcube_aligned_depth,-static_cast<int>(margin_width_0),-static_cast<int>(margin_height_0),-static_cast<int>(margin_depth_0));

        fftwf_execute_dft_r2c(m_fft_plan_psf_r2c,m_h_psf_aligned,reinterpret_cast<fftwf_complex*>(m_h_psf_aligned_fftw3));
    }

    {
        std::size_t intcube_length = m_intcube_depth*m_intcube_height*m_intcube_width;
        std::size_t intcube_aligned_length = m_intcube_aligned_depth*m_intcube_aligned_height*m_intcube_aligned_width;
        std::size_t intcube_aligned_fft_length = m_intcube_aligned_depth*m_intcube_aligned_height*(m_intcube_aligned_width/2+1);

        m_h_intcube = reinterpret_cast<float*>(fftwf_malloc(intcube_length*sizeof(float)));
        m_h_intcube_aligned = reinterpret_cast<float*>(fftwf_malloc(intcube_aligned_length*sizeof(float)));
        m_h_intcube_aligned_fftw3 = reinterpret_cast<std::complex<float>*>(fftwf_malloc(intcube_aligned_fft_length*sizeof(std::complex<float>)));

        m_fft_plan_intcube_r2c = fftwf_plan_dft_r2c_3d(m_intcube_aligned_depth,m_intcube_aligned_height,m_intcube_aligned_width,m_h_intcube_aligned,reinterpret_cast<fftwf_complex*>(m_h_intcube_aligned_fftw3),FFTW_ESTIMATE);
        m_fft_plan_intcube_c2r = fftwf_plan_dft_c2r_3d(m_intcube_aligned_depth,m_intcube_aligned_height,m_intcube_aligned_width,reinterpret_cast<fftwf_complex*>(m_h_intcube_aligned_fftw3),m_h_intcube_aligned,FFTW_ESTIMATE);

        std::fill(m_h_intcube,m_h_intcube+intcube_length,0);
        std::fill(m_h_intcube_aligned,m_h_intcube_aligned+intcube_aligned_length,0);
        std::fill(m_h_intcube_aligned_fftw3,m_h_intcube_aligned_fftw3+intcube_aligned_fft_length,std::complex<float>(0,0));
    }

    {
        std::size_t velsigmap_length = m_intcube_height*m_intcube_width;
        std::size_t velsigmap_aligned_length = m_intcube_aligned_height*m_intcube_aligned_width;

        m_h_velmap = reinterpret_cast<float*>(fftwf_malloc(velsigmap_length*sizeof(float)));
        m_h_sigmap = reinterpret_cast<float*>(fftwf_malloc(velsigmap_length*sizeof(float)));
        m_h_velmap_aligned = reinterpret_cast<float*>(fftwf_malloc(velsigmap_aligned_length*sizeof(float)));
        m_h_sigmap_aligned = reinterpret_cast<float*>(fftwf_malloc(velsigmap_aligned_length*sizeof(float)));

        std::fill(m_h_velmap,m_h_velmap+velsigmap_length,0);
        std::fill(m_h_sigmap,m_h_sigmap+velsigmap_length,0);
        std::fill(m_h_velmap_aligned,m_h_velmap_aligned+velsigmap_aligned_length,0);
        std::fill(m_h_sigmap_aligned,m_h_sigmap_aligned+velsigmap_aligned_length,0);
    }

}

model_thindisk::~model_thindisk()
{
    fftwf_free(m_h_intcube);
    fftwf_free(m_h_intcube_aligned);
    fftwf_free(m_h_intcube_aligned_fftw3);

    fftwf_free(m_h_psf);
    fftwf_free(m_h_psf_aligned);
    fftwf_free(m_h_psf_aligned_fftw3);

    fftwf_free(m_h_velmap);
    fftwf_free(m_h_sigmap);

    fftwf_free(m_h_velmap_aligned);
    fftwf_free(m_h_sigmap_aligned);

    fftwf_destroy_plan(m_fft_plan_psf_r2c);
    fftwf_destroy_plan(m_fft_plan_intcube_r2c);
    fftwf_destroy_plan(m_fft_plan_intcube_c2r);

    fftwf_cleanup_threads();
}

std::size_t model_thindisk::get_model_data_length(void) const
{
    return m_intcube_width*m_intcube_height*2;
}

void model_thindisk::evaluate(unsigned int model_id,
                              const std::vector<float>& model_params_proj,
                              const std::vector<float>& model_params_flux,
                              const std::vector<float>& model_params_rcur,
                              const std::vector<float>& model_params_vsig,
                              float vsys,
                              std::valarray<float>& model_data)
{
    // cube sampling
    std::vector<float> cube_sampling;
    cube_sampling.push_back(m_step_x/m_upsampling_x);
    cube_sampling.push_back(m_step_y/m_upsampling_y);
    cube_sampling.push_back(m_step_z/m_upsampling_z);

    // this is the margin required in order to deal with the incorect values at the edges caused by the psf convolution
    int margin_width_0, margin_width_1, margin_height_0, margin_height_1, margin_depth_0, margin_depth_1;
    get_intcube_margins(margin_width_0,margin_width_1,margin_height_0,margin_height_1,margin_depth_0,margin_depth_1);

    // evaluate high resolution model cube
    kernels_omp::model_image_3d_evaluate(m_h_intcube_aligned,
                                         model_id,
                                         m_upsampling_x*m_intcube_width,
                                         m_upsampling_y*m_intcube_height,
                                         m_upsampling_z*m_intcube_depth,
                                         m_intcube_aligned_width,
                                         m_intcube_aligned_height,
                                         m_intcube_aligned_depth,
                                         margin_width_0,
                                         margin_width_1,
                                         margin_height_0,
                                         margin_height_1,
                                         margin_depth_0,
                                         margin_depth_1,
                                         model_params_proj.data(),
                                         model_params_proj.size(),
                                         model_params_flux.data(),
                                         model_params_flux.size(),
                                         model_params_rcur.data(),
                                         model_params_rcur.size(),
                                         model_params_vsig.data(),
                                         model_params_vsig.size(),
                                         cube_sampling.data(),
                                         cube_sampling.size());

    // convolve with psf
    if(false)
        kernels_omp::model_image_3d_convolve_fft(m_h_intcube_aligned,
                                                 m_h_intcube_aligned_fftw3,
                                                 m_h_psf_aligned_fftw3,
                                                 m_intcube_aligned_width,
                                                 m_intcube_aligned_height,
                                                 m_intcube_aligned_depth,
                                                 1,
                                                 m_fft_plan_intcube_r2c,
                                                 m_fft_plan_intcube_c2r);

    // downsample high resolution cube
    kernels_omp::model_image_3d_downsample(m_h_intcube,
                                           m_h_intcube_aligned,
                                           m_intcube_width,
                                           m_intcube_height,
                                           m_intcube_depth,
                                           m_intcube_aligned_width,
                                           m_intcube_aligned_height,
                                           m_intcube_aligned_depth,
                                           margin_width_0,
                                           margin_height_0,
                                           margin_depth_0,
                                           m_upsampling_x,
                                           m_upsampling_y,
                                           m_upsampling_z);

    // extract velmap and sigmap from low resolution cube
    kernels_omp::model_image_3d_extract_moment_maps(m_h_velmap,
                                                    m_h_sigmap,
                                                    m_h_intcube,
                                                    m_intcube_width,
                                                    m_intcube_height,
                                                    m_intcube_depth,
                                                    cube_sampling.data(),
                                                    cube_sampling.size(),
                                                    vsys,
                                                    0);

    // extract velmap and sigmap from low resolution cube
    kernels_omp::model_image_3d_extract_moment_maps(m_h_velmap_aligned,
                                                    m_h_sigmap_aligned,
                                                    m_h_intcube_aligned,
                                                    m_intcube_aligned_width,
                                                    m_intcube_aligned_height,
                                                    m_intcube_aligned_depth,
                                                    cube_sampling.data(),
                                                    cube_sampling.size(),
                                                    vsys,
                                                    0);

    /*
    // copy velmap and sigmap to output
    model_data.resize(get_model_data_length());
    std::size_t map_length = m_intcube_height*m_intcube_width;
    std::copy(m_h_velmap,m_h_velmap+map_length,std::begin(model_data));
    std::copy(m_h_sigmap,m_h_sigmap+map_length,std::begin(model_data)+map_length);
    */
    std::size_t map_length = m_intcube_height*m_intcube_width;
    std::size_t map_length_aligned = m_intcube_aligned_height*m_intcube_aligned_width;
    std::size_t cube_length = m_intcube_height*m_intcube_width*m_intcube_depth;
    std::size_t cube_length_aligned = m_intcube_aligned_height*m_intcube_aligned_width*m_intcube_aligned_depth;

    model_data.resize(map_length*2+map_length_aligned*2+cube_length+cube_length_aligned);

    std::size_t offset = 0;
    std::copy(m_h_velmap,m_h_velmap+map_length,std::begin(model_data)+offset);
    offset += map_length;
    std::copy(m_h_sigmap,m_h_sigmap+map_length,std::begin(model_data)+offset);
    offset += map_length;
    std::copy(m_h_velmap_aligned,m_h_velmap_aligned+map_length_aligned,std::begin(model_data)+offset);
    offset += map_length_aligned;
    std::copy(m_h_sigmap_aligned,m_h_sigmap_aligned+map_length_aligned,std::begin(model_data)+offset);
    offset += map_length_aligned;
    std::copy(m_h_intcube,m_h_intcube+cube_length,std::begin(model_data)+offset);
    offset += cube_length;
    std::copy(m_h_intcube_aligned,m_h_intcube_aligned+cube_length_aligned,std::begin(model_data)+offset);
    offset += cube_length_aligned;
}

void model_thindisk::get_intcube_margins(int& width_0, int& width_1, int& height_0, int& height_1, int& depth_0, int& depth_1) const
{
    width_0 = width_1 = height_0 = height_1 = depth_0 = depth_1 = 0;

    if (m_psf_width > 0 && m_psf_height > 0 && m_psf_depth > 0)
    {
        width_0 = m_psf_width/2;
        width_1 = m_psf_width - width_0 - 1;
        height_0 = m_psf_height/2;
        height_1 = m_psf_height - height_0 - 1;
        depth_0 = m_psf_depth/2;
        depth_1 = m_psf_depth - depth_0 - 1;
    }
}

void model_thindisk::get_intcube_aligned_lengths(std::size_t& width, std::size_t& height, std::size_t& depth) const
{
    width = m_intcube_aligned_width;
    height = m_intcube_aligned_height;
    depth = m_intcube_aligned_depth;
}

}
}
