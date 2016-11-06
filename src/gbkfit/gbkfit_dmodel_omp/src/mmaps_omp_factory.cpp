
#include "gbkfit/dmodel/mmaps/mmaps_omp_factory.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_omp.hpp"

#include "gbkfit/json.hpp"
#include "gbkfit/utility.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

const std::string MMapsOmpFactory::FACTORY_TYPE = "gbkfit.dmodel.mmaps_omp";

const std::string& MMapsOmpFactory::get_type(void) const
{
    return FACTORY_TYPE;
}

DModel* MMapsOmpFactory::create(const std::string& info,
                                const std::vector<int>& size,
                                const std::vector<float>& step,
                                const PointSpreadFunction* psf,
                                const LineSpreadFunction* lsf) const
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::vector<int> size_ = size;
    std::vector<float> step_ = step;
    std::vector<int> upsampling;

    MomentMethod method = MomentMethod::moments;
    std::string method_str = info_root.value("method", "moments");
    if (method_str == "gaussian")
        method = MomentMethod::gaussian;
    else if (method_str == "hermite")
        method = MomentMethod::hermite;
    else
        method = MomentMethod::moments;

    if (size_.empty()) {
        size_ = info_root.at("size").get<std::vector<int>>();
    }

    if (step_.empty()) {
        step_ = info_root.at("step").get<std::vector<float>>();
    }

    if (size_.size() == 2) {
        int sz = util_num::roundu_odd((2*500+200)/step_[2]);
        size_.push_back(sz);
    }

    upsampling = info_root.at("upsampling").get<std::vector<int>>();

    return new MMapsOmp(size_[0],
                        size_[1],
                        size_[2],
                        step_[0],
                        step_[1],
                        step_[2],
                        upsampling[0],
                        upsampling[1],
                        upsampling[2],
                        method,
                        psf,
                        lsf);
}

void MMapsOmpFactory::destroy(DModel* model) const
{
    if (model->get_type() != get_type()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    delete model;
}

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
