#pragma once
#ifndef GBKFIT_CORE_HPP
#define GBKFIT_CORE_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

class Core final
{

private:

    std::map<std::string, const FitterFactory*> m_fitter_factories;

    std::map<std::string, const DModelFactory*> m_dmodel_factories;
    std::map<std::string, const GModelFactory*> m_gmodel_factories;

    std::vector<Dataset*> m_datasets;
    std::vector<PointSpreadFunction*> m_psfs;
    std::vector<LineSpreadFunction*> m_lsfs;
    std::vector<Fitter*> m_fitters;
    std::vector<Params*> m_parameters;

    std::vector<DModel*> m_dmodels;
    std::vector<GModel*> m_gmodels;

public:

    Core(void);

    ~Core();

    void add_dmodel_factory(const DModelFactory* factory);

    void add_gmodel_factory(const GModelFactory* factory);

    void add_fitter_factory(const FitterFactory* factory);

    std::vector<Dataset*> create_datasets(const std::string& info);

    Dataset* create_dataset(const std::string& info);

    Fitter* create_fitter(const std::string& info);

    Params* create_parameters(const std::string& info);

    DModel* create_dmodel(const std::string& info,
                          const std::vector<int>& size,
                          const std::vector<float>& step,
                          const PointSpreadFunction* psf,
                          const LineSpreadFunction* lsf);

    GModel* create_gmodel(const std::string& info);


    LineSpreadFunction* create_line_spread_function(const std::string& info);

    PointSpreadFunction* create_point_spread_function(const std::string& info);

private:

    const FitterFactory* get_fitter_factory(const std::string& type) const;

    const DModelFactory* get_dmodel_factory(const std::string& type) const;

    const GModelFactory* get_gmodel_factory(const std::string& type) const;

};

} // namespace gbkfit

#endif // GBKFIT_CORE_HPP
