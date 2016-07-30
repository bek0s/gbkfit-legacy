#pragma once
#ifndef GBKFIT_DMODEL_MMAPS_MMAPS_OMP_FACTORY_HPP
#define GBKFIT_DMODEL_MMAPS_MMAPS_OMP_FACTORY_HPP

#include "gbkfit/dmodel/mmaps/mmaps_omp.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_factory.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

class MMapsOmpFactory : public MMapsFactory
{

public:

    static const std::string FACTORY_TYPE;

    MMapsOmpFactory(void) {}

    ~MMapsOmpFactory() {}

    const std::string& get_type(void) const override final;

    DModel* create(const std::string& info,
                   const std::vector<int>& size,
                   const std::vector<float>& step,
                   const PointSpreadFunction* psf,
                   const LineSpreadFunction* lsf) const override final;

    void destroy(DModel* model) const override final;

};

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_MMAPS_MMAPS_OMP_FACTORY_HPP
