#pragma once
#ifndef GBKFIT_APP_CLI_APPLICATION_HPP
#define GBKFIT_APP_CLI_APPLICATION_HPP

#include "prerequisites.hpp"
#include "gbkfit/core.hpp"

namespace gbkfit_app_cli {

//!
//! \brief The Application class
//!
class Application
{

private:

    static const std::string DEFAULT_CONFIG_FILE;
    static const std::string DEFAULT_GALAXY_NAME;

private:

    std::string m_config_file;
    std::string m_galaxy_name;

    gbkfit::Core* m_core;

    std::vector<gbkfit::ModelFactory*> m_model_factories;
    std::vector<gbkfit::FitterFactory*> m_fitter_factories;

    gbkfit::Model* m_model;
    gbkfit::Fitter* m_fitter;
    gbkfit::Parameters* m_parameters;

    std::map<std::string,gbkfit::Dataset*> m_datasets;

    gbkfit::instrument* m_instrument;

public:

    Application(void);

    ~Application();

    bool process_program_options(int argc, char** argv);

    bool initialize(void);

    void shutdown(void);

    void run(void);

}; // class Application

} // namespace gbkfit_app_cli

#endif // GBKFIT_APP_CLI_APPLICATION_HPP
