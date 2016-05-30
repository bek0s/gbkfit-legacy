#pragma once
#ifndef GBKFIT_DMODEL_SCUBE_SCUBE_HPP
#define GBKFIT_DMODEL_SCUBE_SCUBE_HPP

#include "gbkfit/dmodel.hpp"

namespace gbkfit {
namespace dmodel {
namespace scube {

class SCube : public DModel
{

protected:

    const GModel* m_gmodel;

public:

    SCube(void);

    const GModel* get_galaxy_model(void) const override final;

    void set_galaxy_model(const GModel* gmodel) override final;

};

} // namespace scube
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_SCUBE_SCUBE_HPP
