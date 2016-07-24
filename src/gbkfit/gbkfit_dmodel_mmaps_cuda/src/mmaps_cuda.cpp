
#include "gbkfit/dmodel/mmaps/mmaps_cuda.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_cuda_factory.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_cuda_kernels_h.hpp"

#include "gbkfit/cuda/ndarray.hpp"

#include "gbkfit/instrument.hpp"
#include "gbkfit/ndarray_host.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {


MMapsCuda::MMapsCuda(int size_x,
                     int size_y,
                     float step_x,
                     float step_y,
                     const Instrument* instrument)
    : MMapsCuda(size_x, size_y, step_x, step_y, 1, 1, instrument)
{
}

MMapsCuda::MMapsCuda(int size_x,
                     int size_y,
                     float step_x,
                     float step_y,
                     int upsampling_x,
                     int upsampling_y,
                     const Instrument* instrument)
{
    //
    // Create spectral cube from which the moment maps are derived
    //

    m_scube = new scube::SCubeCuda(size_x,
                                   size_y,
                                   151,
                                   step_x,
                                   step_y,
                                   10,
                                   upsampling_x,
                                   upsampling_y,
                                   1,
                                   instrument);

    //
    // Allocate and initialize moment maps
    //

    int size = size_x*size_y;

    m_h_flxmap = new NDArrayHost({size_x, size_y});
    m_h_velmap = new NDArrayHost({size_x, size_y});
    m_h_sigmap = new NDArrayHost({size_x, size_y});

    m_d_flxmap = new cuda::NDArrayManaged({size_x, size_y});
    m_d_velmap = new cuda::NDArrayManaged({size_x, size_y});
    m_d_sigmap = new cuda::NDArrayManaged({size_x, size_y});

    std::fill_n(m_h_flxmap->get_host_ptr(), size, -1);
    std::fill_n(m_h_velmap->get_host_ptr(), size, -1);
    std::fill_n(m_h_sigmap->get_host_ptr(), size, -1);

    m_d_flxmap->write_data(m_h_flxmap->get_host_ptr());
    m_d_velmap->write_data(m_h_velmap->get_host_ptr());
    m_d_sigmap->write_data(m_h_sigmap->get_host_ptr());

    //
    // Add output data to the output data map
    //

    m_h_output_map["flxmap"] = m_h_flxmap;
    m_h_output_map["velmap"] = m_h_velmap;
    m_h_output_map["sigmap"] = m_h_sigmap;

    m_d_output_map["flxmap"] = m_d_flxmap;
    m_d_output_map["velmap"] = m_d_velmap;
    m_d_output_map["sigmap"] = m_d_sigmap;
}

MMapsCuda::~MMapsCuda()
{
    delete m_scube;
    delete m_h_flxmap;
    delete m_h_velmap;
    delete m_h_sigmap;
    delete m_d_flxmap;
    delete m_d_velmap;
    delete m_d_sigmap;
}

const std::string& MMapsCuda::get_type(void) const
{
    return MMapsCudaFactory::FACTORY_TYPE;
}

const std::vector<int>& MMapsCuda::get_size(void) const
{
    return m_scube->get_size();
}

const std::vector<float>& MMapsCuda::get_step(void) const
{
    return m_scube->get_step();
}

const GModel* MMapsCuda::get_galaxy_model(void) const
{
    return m_scube->get_galaxy_model();
}

void MMapsCuda::set_galaxy_model(const GModel* gmodel)
{
    m_scube->set_galaxy_model(gmodel);
}

const std::map<std::string, NDArrayHost*>& MMapsCuda::evaluate(
        const std::map<std::string, float>& params) const
{
    evaluate_managed(params);

    m_d_flxmap->read_data(m_h_flxmap->get_host_ptr());
    m_d_velmap->read_data(m_h_velmap->get_host_ptr());
    m_d_sigmap->read_data(m_h_sigmap->get_host_ptr());

    return m_h_output_map;
}

const std::map<std::string, cuda::NDArrayManaged*>& MMapsCuda::evaluate_managed(
        const std::map<std::string, float>& params) const
{
    const cuda::NDArrayManaged* cube_data = m_scube->evaluate_managed(params).at("flxcube");

    NDShape cube_shape = cube_data->get_shape();
    int size_x = cube_shape[0];
    int size_y = cube_shape[1];
    int size_z = cube_shape[2];

    // ...
//  float step_z = get_instrument()->get_step_z();
    //OMG
    float step_z = 10;
    float zero_z = -size_z/2 * step_z;

    kernels_cuda_h::extract_maps_mmnt(cube_data->get_cuda_ptr(),
                                      size_x,
                                      size_y,
                                      size_z,
                                      zero_z,
                                      step_z,
                                      m_d_flxmap->get_cuda_ptr(),
                                      m_d_velmap->get_cuda_ptr(),
                                      m_d_sigmap->get_cuda_ptr());

    return m_d_output_map;
}

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
