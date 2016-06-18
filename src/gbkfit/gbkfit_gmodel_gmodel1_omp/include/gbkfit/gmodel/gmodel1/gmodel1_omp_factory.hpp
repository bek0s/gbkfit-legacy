#pragma once
#ifndef GBKFIT_GMODEL_GMODEL1_GMODEL1_OMP_FACTORY_HPP
#define GBKFIT_GMODEL_GMODEL1_GMODEL1_OMP_FACTORY_HPP

#include "gbkfit/gmodel/gmodel1/gmodel1_factory.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel1 {

class GModel1OmpFactory : public GModel1Factory
{

public:

    static const std::string FACTORY_TYPE;

    GModel1OmpFactory(void) {}

    ~GModel1OmpFactory() {}

    const std::string& get_type(void) const override final;

    GModel* create(const std::string& info) const override final;

    void destroy(GModel* gmodel) const override final;

};

} // namespace gmodel1
} // namespace gmodel
} // namespace gbkfit

#endif // GBKFIT_GMODEL_GMODEL1_GMODEL1_OMP_FACTORY_HPP
