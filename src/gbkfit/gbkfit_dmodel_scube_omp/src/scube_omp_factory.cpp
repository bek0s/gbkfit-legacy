
#include "gbkfit/dmodel/scube/scube_omp_factory.hpp"
#include "gbkfit/dmodel/scube/scube_omp.hpp"

#include "gbkfit/json.hpp"

namespace gbkfit {
namespace dmodel {
namespace scube {

const std::string SCubeOmpFactory::FACTORY_TYPE = "gbkfit.dmodel.scube_omp";

const std::string& SCubeOmpFactory::get_type(void) const
{
    return FACTORY_TYPE;
}

DModel* SCubeOmpFactory::create(const std::string& info,
                                const std::vector<int>& shape,
                                const Instrument* instrument) const
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    int size_x, size_y, size_z;

    if (shape.size() == 3)
    {
        size_x = shape[0];
        size_y = shape[1];
        size_z = shape[2];
    }
    else
    {
        size_x = info_root.at("size_x").get<int>();
        size_y = info_root.at("size_y").get<int>();
        size_z = info_root.at("size_z").get<int>();
    }

    return new SCubeOmp(size_x, size_y, size_z, instrument);
}

void SCubeOmpFactory::destroy(DModel* dmodel) const
{
    if (dmodel->get_type() == get_type()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    delete dmodel;
}

} // namespace scube
} // namespace dmodel
} // namespace gbkfit
