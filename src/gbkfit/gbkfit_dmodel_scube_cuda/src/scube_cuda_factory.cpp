
#include "gbkfit/dmodel/scube/scube_cuda_factory.hpp"
#include "gbkfit/dmodel/scube/scube_cuda.hpp"

#include "gbkfit/json.hpp"

namespace gbkfit {
namespace dmodel {
namespace scube {

const std::string SCubeCudaFactory::FACTORY_TYPE = "gbkfit.dmodel.scube_cuda";

const std::string& SCubeCudaFactory::get_type(void) const
{
    return FACTORY_TYPE;
}

DModel* SCubeCudaFactory::create(const std::string& info,
                                 const std::vector<int>& size,
                                 const std::vector<float>& step,
                                 const Instrument* instrument) const
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    int size_x, size_y, size_z;
    float step_x, step_y, step_z;

    if (size.size() == 3)
    {
        size_x = size[0];
        size_y = size[1];
        size_z = size[2];
    }
    else
    {
        size_x = info_root.at("size").at(0).get<int>();
        size_y = info_root.at("size").at(1).get<int>();
        size_z = info_root.at("size").at(2).get<int>();
    }

    if (step.size() == 3)
    {
        step_x = step[0];
        step_y = step[1];
        step_z = step[2];
    }
    else
    {
        step_x = info_root.at("step").at(0).get<float>();
        step_y = info_root.at("step").at(1).get<float>();
        step_z = info_root.at("step").at(2).get<float>();
    }

    return new SCubeCuda(size_x, size_y, size_z,
                         step_x, step_y, step_z,
                         instrument);
}

void SCubeCudaFactory::destroy(DModel* dmodel) const
{
    if (dmodel->get_type() != get_type())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    delete dmodel;
}

} // namespace scube
} // namespace dmodel
} // namespace gbkfit
