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

    gbkfit::core* m_core;

    gbkfit::fitter_factory* m_fitter_factory_mpfit;
    gbkfit::fitter_factory* m_fitter_factory_multinest;
    gbkfit::model_factory* m_model_factory_galaxy_2d_cuda;
    gbkfit::model_factory* m_model_factory_galaxy_2d_omp;
    gbkfit::fitter* m_fitter;
    gbkfit::model* m_model;

public:

    application(void);

    ~application();

    bool initialize(void);

    void shutdown(void);

    void run(void);

}; // class application


} // namespace gbkfit_app_cli

#endif // GBKFIT_APP_CLI_APPLICATION_HPP
