
#include "gbkfit/dmodel/mmaps/mmaps_cuda_factory.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_cuda.hpp"

#include "gbkfit/json.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

const std::string MMapsCudaFactory::FACTORY_TYPE = "gbkfit.dmodel.mmaps_cuda";

const std::string& MMapsCudaFactory::get_type(void) const
{
    return FACTORY_TYPE;
}

DModel* MMapsCudaFactory::create(const std::string& info,
                                 const std::vector<int>& size,
                                 const std::vector<float>& step,
                                 const Instrument* instrument) const
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    int size_x, size_y;
    float step_x, step_y;

    if (size.size() == 2)
    {
        size_x = size[0];
        size_y = size[1];
    }
    else
    {
        size_x = info_root.at("size").at(0).get<int>();
        size_y = info_root.at("size").at(1).get<int>();
    }

    if (step.size() == 2)
    {
        step_x = step[0];
        step_y = step[1];
    }
    else
    {
        step_x = info_root.at("step").at(0).get<float>();
        step_y = info_root.at("step").at(1).get<float>();
    }

    return new MMapsCuda(size_x,
                         size_y,
                         step_x,
                         step_y,
                         instrument);
}

void MMapsCudaFactory::destroy(DModel* dmodel) const
{
    if (dmodel->get_type() != get_type())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    delete dmodel;
}

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
