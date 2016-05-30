#pragma once
#ifndef GBKFIT_GMODEL_GMODEL01_GMODEL01_CUDA_FACTORY_HPP
#define GBKFIT_GMODEL_GMODEL01_GMODEL01_CUDA_FACTORY_HPP

#include "gbkfit/gmodel/gmodel01/gmodel01_factory.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel01 {

class GModel01CudaFactory : public GModel01Factory
{

public:

    static const std::string FACTORY_TYPE;

    GModel01CudaFactory(void) {}

    ~GModel01CudaFactory() {}

    const std::string& get_type(void) const override final;

    GModel* create(const std::string& info) const override final;

    void destroy(GModel* gmodel) const override final;

};

} // namespace gmodel01
} // namespace gmodel
} // namespace gbkfit

#endif // GBKFIT_GMODEL_GMODEL01_GMODEL01_CUDA_FACTORY_HPP
