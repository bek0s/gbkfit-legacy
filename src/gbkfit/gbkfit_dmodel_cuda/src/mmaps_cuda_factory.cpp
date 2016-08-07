
#include "gbkfit/dmodel/mmaps/mmaps_cuda_factory.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_cuda.hpp"

#include "gbkfit/json.hpp"
#include "gbkfit/utility.hpp"

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
                                 const PointSpreadFunction* psf,
                                 const LineSpreadFunction* lsf) const
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::vector<int> size_ = size;
    std::vector<float> step_ = step;
    std::vector<int> upsampling;

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

    return new MMapsCuda(size_[0],
                         size_[1],
                         size_[2],
                         step_[0],
                         step_[1],
                         step_[2],
                         upsampling[0],
                         upsampling[1],
                         upsampling[2],
                         psf,
                         lsf);
}

void MMapsCudaFactory::destroy(DModel* dmodel) const
{
    if (dmodel->get_type() != get_type()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    delete dmodel;
}

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
