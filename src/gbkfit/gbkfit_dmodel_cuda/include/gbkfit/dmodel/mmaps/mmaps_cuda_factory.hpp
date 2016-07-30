#pragma once
#ifndef GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_FACTORY_HPP
#define GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_FACTORY_HPP

#include "gbkfit/dmodel.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

class MMapsCudaFactory : public DModelFactory
{

public:

    static const std::string FACTORY_TYPE;

    MMapsCudaFactory(void) {}

    ~MMapsCudaFactory() {}

    const std::string& get_type(void) const override final;

    DModel* create(const std::string& info,
                   const std::vector<int>& size,
                   const std::vector<float>& step,
                   const PointSpreadFunction* psf,
                   const LineSpreadFunction* lsf) const override final;

    void destroy(DModel* dmodel) const override final;

};

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_FACTORY_HPP
