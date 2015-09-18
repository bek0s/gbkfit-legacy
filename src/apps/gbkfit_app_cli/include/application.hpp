#pragma once
#ifndef GBKFIT_APP_CLI_APPLICATION_HPP
#define GBKFIT_APP_CLI_APPLICATION_HPP

#include "prerequisites.hpp"
#include "gbkfit/core.hpp"

namespace gbkfit_app_cli {

//!
//! \brief The application class
//!
class application
{

private:

    std::string m_filename_config;

    gbkfit::core* m_core;

    gbkfit::model_factory* m_model_factory_model01_cuda;
    gbkfit::model_factory* m_model_factory_model01_omp;

    gbkfit::fitter_factory* m_fitter_factory_mpfit;
    gbkfit::fitter_factory* m_fitter_factory_multinest;

    gbkfit::model* m_model;
    gbkfit::fitter* m_fitter;
    gbkfit::parameters_fit_info* m_fit_info;

    std::map<std::string,gbkfit::nddataset*> m_datasets;

    gbkfit::instrument* m_instrument;

public:

    application(void);

    ~application();

    bool initialize(void);

    void shutdown(void);

    void run(void);

}; // class application

} // namespace gbkfit_app_cli

#endif // GBKFIT_APP_CLI_APPLICATION_HPP
