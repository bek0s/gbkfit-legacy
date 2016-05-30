
#include "gbkfit/gmodel/gmodel01/gmodel01_cuda_factory.hpp"
#include "gbkfit/gmodel/gmodel01/gmodel01_cuda.hpp"

#include "gbkfit/json.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel01 {

const std::string GModel01CudaFactory::FACTORY_TYPE =
        "gbkfit.gmodel.gmodel01_cuda";

const std::string& GModel01CudaFactory::get_type(void) const
{
    return FACTORY_TYPE;
}

GModel* GModel01CudaFactory::create(const std::string& info) const
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::string flx_profile_name = info_root.at("flx_profile")
                                            .get<std::string>();
    std::string vel_profile_name = info_root.at("vel_profile")
                                            .get<std::string>();

    FlxProfileType flx_profile;
    VelProfileType vel_profile;

    if (flx_profile_name == "exponential")
        flx_profile = FlxProfileType::exponential;
    else
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    if      (vel_profile_name == "lramp")
        vel_profile = VelProfileType::lramp;
    else if (vel_profile_name == "boissier")
        vel_profile = VelProfileType::boissier;
    else if (vel_profile_name == "arctan")
        vel_profile = VelProfileType::arctan;
    else if (vel_profile_name == "epinat")
        vel_profile = VelProfileType::epinat;
    else
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    return new GModel01Cuda(flx_profile, vel_profile);
}

void GModel01CudaFactory::destroy(GModel* gmodel) const
{
    if (gmodel->get_type() != get_type()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    delete gmodel;
}

} // namespace gmodel01
} // namespace gmodel
} // namespace gbkfit
