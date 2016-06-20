#pragma once
#ifndef GBKFIT_APP_CLI_APPLICATION_HPP
#define GBKFIT_APP_CLI_APPLICATION_HPP

#include "gbkfit/prerequisites.hpp"

#include "prerequisites.hpp"

namespace gbkfit_app_cli {

class Application
{

private:

    static const std::string DEFAULT_CONFIG_FILE;
    static const std::string DEFAULT_OUTPUT_DIR;

private:

    std::string m_config_file;
    std::string m_output_dir;

    std::string m_task;

    gbkfit::Core* m_core;

    std::vector<gbkfit::DModelFactory*> m_dmodel_factories;
    std::vector<gbkfit::GModelFactory*> m_gmodel_factories;
    std::vector<gbkfit::FitterFactory*> m_fitter_factories;

    gbkfit::DModel* m_dmodel;
    gbkfit::GModel* m_gmodel;
    gbkfit::Fitter* m_fitter;
    gbkfit::Params* m_params;
    std::vector<gbkfit::Dataset*> m_datasets;
    gbkfit::Instrument* m_instrument;

public:

    Application(void);

    ~Application();

    bool process_program_options(int argc, char** argv);

    bool initialize(void);

    void shutdown(void);

    void run(void);

};

} // namespace gbkfit_app_cli

#endif // GBKFIT_APP_CLI_APPLICATION_HPP
