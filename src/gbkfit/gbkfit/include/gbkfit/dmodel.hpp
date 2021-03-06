#pragma once
#ifndef GBKFIT_DMODEL_HPP
#define GBKFIT_DMODEL_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

class DModel
{

public:

    DModel(void);

    virtual ~DModel();

    virtual const std::string& get_type(void) const = 0;

    virtual const std::vector<int>& get_size(void) const = 0;

    virtual const std::vector<float>& get_step(void) const = 0;

    virtual const GModel* get_galaxy_model(void) const = 0;

    virtual void set_galaxy_model(const GModel* gmodel) = 0;

    virtual const std::map<std::string, NDArrayHost*>& evaluate(
            const std::map<std::string, float>& params) const = 0;

};

class DModelFactory
{

public:

    DModelFactory(void);

    virtual ~DModelFactory();

    virtual const std::string& get_type(void) const = 0;

    virtual DModel* create(const std::string& info,
                           const std::vector<int>& size,
                           const std::vector<float>& step,
                           const PointSpreadFunction* psf,
                           const LineSpreadFunction* lsf) const = 0;

    virtual void destroy(DModel* dmodel) const = 0;

};

} // namespace gbkfit

#endif // GBKFIT_DMODEL_HPP
