#pragma once
#ifndef GBKFIT_GMODEL_HPP
#define GBKFIT_GMODEL_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

class GModel
{

public:

    GModel(void);

    virtual ~GModel();

    virtual const std::string& get_type(void) const = 0;

    virtual const std::vector<std::string>& get_param_names(void) const = 0;

    virtual void evaluate(const std::map<std::string, float>& params,
                          const std::vector<float>& data_zero,
                          const std::vector<float>& data_step,
                          NDArray* data) const = 0;

};

class GModelFactory
{

public:

    GModelFactory(void);

    virtual ~GModelFactory();

    virtual const std::string& get_type(void) const = 0;

    virtual GModel* create(const std::string& info) const = 0;

    virtual void destroy(GModel* model) const = 0;

};

} // namespace gbkfit

#endif // GBKFIT_GMODEL_HPP
