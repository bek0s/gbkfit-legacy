
#include "gbkfit/dmodel/mmaps/mmaps_omp.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_omp_factory.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_omp_kernels.hpp"

#include "gbkfit/instrument.hpp"
#include "gbkfit/ndarray_host.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

MMapsOmp::MMapsOmp(int size_x,
                   int size_y,
                   float step_x,
                   float step_y,
                   const Instrument *instrument)
    : MMapsOmp(size_x,
               size_y,
               step_x,
               step_y,
               1,
               1,
               instrument)
{
}

MMapsOmp::MMapsOmp(int size_x,
                   int size_y,
                   float step_x,
                   float step_y,
                   int upsampling_x,
                   int upsampling_y,
                   const Instrument *instrument)
{
    //
    // Create spectral cube from which the moment maps are derived
    //

    m_scube = new scube::SCubeOmp(size_x,
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

    m_flxmap = new NDArrayHost({size_x, size_y});
    m_velmap = new NDArrayHost({size_x, size_y});
    m_sigmap = new NDArrayHost({size_x, size_y});

    std::fill_n(m_flxmap->get_host_ptr(), size, -1);
    std::fill_n(m_velmap->get_host_ptr(), size, -1);
    std::fill_n(m_sigmap->get_host_ptr(), size, -1);

    //
    // Add output data to the output data map
    //

    m_output_map["flxmap"] = m_flxmap;
    m_output_map["velmap"] = m_velmap;
    m_output_map["sigmap"] = m_sigmap;
}

MMapsOmp::~MMapsOmp()
{
    delete m_scube;

    delete m_flxmap;
    delete m_velmap;
    delete m_sigmap;
}

const std::string& MMapsOmp::get_type(void) const
{
    return MMapsOmpFactory::FACTORY_TYPE;
}

const std::vector<int>& MMapsOmp::get_size(void) const
{
    return m_scube->get_size();
}

const std::vector<float>& MMapsOmp::get_step(void) const
{
    return m_scube->get_step();
}

const GModel* MMapsOmp::get_galaxy_model(void) const
{
    return m_scube->get_galaxy_model();
}

void MMapsOmp::set_galaxy_model(const GModel* gmodel)
{
    m_scube->set_galaxy_model(gmodel);
}

const std::map<std::string, NDArrayHost*>& MMapsOmp::evaluate(
        const std::map<std::string, float>& params) const
{
    const NDArrayHost* cube_data = m_scube->evaluate(params).at("flxcube");

    NDShape cube_shape = cube_data->get_shape();
    int size_x = cube_shape[0];
    int size_y = cube_shape[1];
    int size_z = cube_shape[2];

    //  float step_z = get_instrument()->get_step_z();
    //OMG
    float step_z = 10;
    float zero_z = -size_z/2 * step_z;

    kernels_omp::extract_maps_mmnt(cube_data->get_host_ptr(),
                                   size_x,
                                   size_y,
                                   size_z,
                                   zero_z,
                                   step_z,
                                   m_flxmap->get_host_ptr(),
                                   m_velmap->get_host_ptr(),
                                   m_sigmap->get_host_ptr());

    return m_output_map;
}

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
