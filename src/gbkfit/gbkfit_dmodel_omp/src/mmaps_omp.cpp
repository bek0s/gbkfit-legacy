
#include "gbkfit/dmodel/mmaps/mmaps_omp.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_omp_factory.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_omp_kernels.hpp"

#include "gbkfit/ndarray_host.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

MMapsOmp::MMapsOmp(int size_x,
                   int size_y,
                   int size_z,
                   float step_x,
                   float step_y,
                   float step_z,
                   int upsampling_x,
                   int upsampling_y,
                   int upsampling_z,
                   MomentMethod method,
                   const PointSpreadFunction* psf,
                   const LineSpreadFunction* lsf)
    : m_method(method)
{
    //
    // Create spectral cube from which the moment maps are derived
    //

    m_scube = new scube::SCubeOmp(size_x,
                                  size_y,
                                  size_z,
                                  step_x,
                                  step_y,
                                  step_z,
                                  upsampling_x,
                                  upsampling_y,
                                  upsampling_z,
                                  psf,
                                  lsf);

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

    std::vector<int> cube_shape = m_scube->get_size();
    int size_x = cube_shape[0];
    int size_y = cube_shape[1];
    int size_z = cube_shape[2];

    std::vector<float> cube_step = m_scube->get_step();
    float step_z = cube_step[2];

    float zero_z = -size_z/2 * step_z;

    if (m_method == MomentMethod::moments)
    {
        kernels_omp::extract_maps_mmnt(cube_data->get_host_ptr(),
                                       size_x,
                                       size_y,
                                       size_z,
                                       zero_z,
                                       step_z,
                                       m_flxmap->get_host_ptr(),
                                       m_velmap->get_host_ptr(),
                                       m_sigmap->get_host_ptr());
    }
    else
    {
        kernels_omp::extract_maps_gfit(cube_data->get_host_ptr(),
                                       size_x,
                                       size_y,
                                       size_z,
                                       zero_z,
                                       step_z,
                                       m_flxmap->get_host_ptr(),
                                       m_velmap->get_host_ptr(),
                                       m_sigmap->get_host_ptr());
    }

    return m_output_map;
}

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
