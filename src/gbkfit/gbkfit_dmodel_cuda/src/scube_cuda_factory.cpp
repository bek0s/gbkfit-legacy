
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

    upsampling = info_root.at("upsampling").get<std::vector<int>>();

    return new SCubeCuda(size_[0],
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

void SCubeCudaFactory::destroy(DModel* dmodel) const
{
    if (dmodel->get_type() != get_type()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    delete dmodel;
}

} // namespace scube
} // namespace dmodel
} // namespace gbkfit
