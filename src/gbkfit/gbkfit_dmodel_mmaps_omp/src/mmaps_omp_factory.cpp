
#include "gbkfit/dmodel/mmaps/mmaps_omp_factory.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_omp.hpp"

#include "gbkfit/json.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

const std::string MMapsOmpFactory::FACTORY_TYPE = "gbkfit.dmodel.mmaps_omp";

const std::string& MMapsOmpFactory::get_type(void) const
{
    return FACTORY_TYPE;
}

DModel* MMapsOmpFactory::create(const std::string& info,
                                const std::vector<int>& shape,
                                const Instrument* instrument) const
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    int size_x, size_y;

    if (shape.size() == 2)
    {
        size_x = shape[0];
        size_y = shape[1];
    }
    else
    {
        size_x = info_root.at("size_x").get<int>();
        size_y = info_root.at("size_y").get<int>();
    }

    return new MMapsOmp(size_x, size_y, instrument);
}

void MMapsOmpFactory::destroy(DModel* model) const
{
    if (model->get_type() != get_type())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    delete model;
}

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
