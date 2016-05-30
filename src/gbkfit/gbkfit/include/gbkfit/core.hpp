#pragma once
#ifndef GBKFIT_CORE_HPP
#define GBKFIT_CORE_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

class Core final
{

private:

    std::map<std::string, const ModelFactory*> m_model_factories;
    std::map<std::string, const FitterFactory*> m_fitter_factories;

    std::map<std::string, const DModelFactory*> m_dmodel_factories;
    std::map<std::string, const GModelFactory*> m_gmodel_factories;

    std::vector<Dataset*> m_datasets;
    std::vector<Instrument*> m_instruments;
    std::vector<PointSpreadFunction*> m_psfs;
    std::vector<LineSpreadFunction*> m_lsfs;
    std::vector<Model*> m_models;
    std::vector<Fitter*> m_fitters;
    std::vector<Parameters*> m_parameters;

    std::vector<DModel*> m_dmodels;
    std::vector<GModel*> m_gmodels;

public:

    Core(void);

    ~Core();

    void add_model_factory(const ModelFactory* factory);

    void add_dmodel_factory(const DModelFactory* factory);

    void add_gmodel_factory(const GModelFactory* factory);

    void add_fitter_factory(const FitterFactory* factory);

    std::vector<Dataset*> create_datasets(const std::string& info);

    Dataset* create_dataset(const std::string& info);

    Instrument* create_instrument(const std::string& info);

    Model* create_model(const std::string& info);

    Fitter* create_fitter(const std::string& info);

    Parameters* create_parameters(const std::string& info);

    DModel* create_dmodel(const std::string& info,
                          const std::vector<int>& shape,
                          const Instrument* instrument);

    GModel* create_gmodel(const std::string& info);

private:

    const ModelFactory* get_model_factory(const std::string& type) const;

    const FitterFactory* get_fitter_factory(const std::string& type) const;

    const DModelFactory* get_dmodel_factory(const std::string& type) const;

    const GModelFactory* get_gmodel_factory(const std::string& type) const;

};

} // namespace gbkfit

#endif // GBKFIT_CORE_HPP
