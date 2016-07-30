#pragma once
#ifndef GBKFIT_DMODEL_SCUBE_SCUBE_OMP_FACTORY_HPP
#define GBKFIT_DMODEL_SCUBE_SCUBE_OMP_FACTORY_HPP

#include "gbkfit/dmodel/scube/scube_factory.hpp"

namespace gbkfit {
namespace dmodel {
namespace scube {

class SCubeOmpFactory : public SCubeFactory
{

public:

    static const std::string FACTORY_TYPE;

    SCubeOmpFactory(void) {}

    ~SCubeOmpFactory() {}

    const std::string& get_type(void) const override final;

    DModel* create(const std::string& info,
                   const std::vector<int>& size,
                   const std::vector<float>& step,
                   const PointSpreadFunction* psf,
                   const LineSpreadFunction* lsf) const override final;

    void destroy(DModel* dmodel) const override final;

};

} // namespace scube
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_SCUBE_SCUBE_OMP_FACTORY_HPP
