
#include "application.hpp"

#include "gbkfit/core.hpp"
#include "gbkfit/dataset.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/instrument.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/parameters.hpp"
#include "gbkfit/utility.hpp"

#ifdef GBKFIT_BUILD_MODEL_MODEL01_OMP
#include "gbkfit/models/model01/model_model01_omp.hpp"
#endif

#ifdef GBKFIT_BUILD_MODEL_MODEL01_CUDA
#include "gbkfit/models/model01/model_model01_cuda.hpp"
#endif

#ifdef GBKFIT_BUILD_FITTER_MPFIT
#include "gbkfit/fitters/mpfit/fitter_mpfit.hpp"
#include "gbkfit/fitters/mpfit/fitter_factory_mpfit.hpp"
#endif

#ifdef GBKFIT_BUILD_FITTER_MULTINEST
#include "gbkfit/fitters/multinest/fitter_multinest.hpp"
#include "gbkfit/fitters/multinest/fitter_factory_multinest.hpp"
#endif

#include <boost/program_options.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/json_parser.hpp>


namespace gbkfit_app_cli {

const std::string Application::DEFAULT_CONFIG_FILE = "../../data/configs/gbkfit_config.xml";
const std::string Application::DEFAULT_GALAXY_NAME = "unnamed_galaxy";

Application::Application(void)
    : m_config_file(DEFAULT_CONFIG_FILE)
    , m_core(nullptr)
    , m_model(nullptr)
    , m_fitter(nullptr)
    , m_parameters(nullptr)
    , m_instrument(nullptr)
{
}

Application::~Application()
{
}

bool Application::process_program_options(int argc, char** argv)
{
    boost::program_options::options_description po_desc("options");
    po_desc.add_options()
            ("config", boost::program_options::value<std::string>(&m_config_file), "")
            ("help",                                                               "");

    boost::program_options::variables_map po_vm;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, po_desc), po_vm);
        boost::program_options::notify(po_vm);
    } catch(boost::program_options::error& e) {
        std::cout << "could not parse command-line arguments, error: " << e.what() << std::endl;
        std::cout << po_desc;
        return false;
    }

    if (po_vm.count("help"))
    {
        std::cout << po_desc;
        return false;
    }

    return true;
}


void set_key(const std::string* str)
{

}


bool Application::initialize(void)
{
    std::cout << "initialization started" << std::endl;

    //
    // Create gbkfit core.
    //

    m_core = new gbkfit::Core();

    //
    // Create factories.
    //

    std::cout << "creating model factories..." << std::endl;
    #ifdef GBKFIT_BUILD_MODEL_MODEL01_OMP
    m_model_factories.push_back(new gbkfit::models::model01::model_factory_model01_omp());
    #endif
    #ifdef GBKFIT_BUILD_MODEL_MODEL01_CUDA
    m_model_factories.push_back(new gbkfit::models::model01::model_factory_model01_cuda());
    #endif

    std::cout << "creating fitter factories..." << std::endl;
    #ifdef GBKFIT_BUILD_FITTER_MPFIT
    m_fitter_factories.push_back(new gbkfit::fitters::mpfit::FitterFactoryMpfit());
    #endif
    #ifdef GBKFIT_BUILD_FITTER_MULTINEST
    m_fitter_factories.push_back(new gbkfit::fitters::multinest::FitterFactoryMultinest());
    #endif

    //
    //  Add factories to the gbkfit core.
    //

    std::cout << "registering model factories..." << std::endl;
    for(std::size_t i = 0; i < m_model_factories.size(); ++i) {
        std::cout << "registering model factory '" << m_model_factories[i]->get_type_name() << "'..." << std::endl;
        m_core->add_model_factory(m_model_factories[i]);
    }

    std::cout << "registering fitter factories..." << std::endl;
    for(std::size_t i = 0; i < m_fitter_factories.size(); ++i) {
        std::cout << "registering fitter factory '" << m_fitter_factories[i]->get_type_name() << "'..." << std::endl;
        m_core->add_fitter_factory(m_fitter_factories[i]);
    }

    //
    // Read configuration.
    //

    std::cout << "reading configuration..." << std::endl;
    boost::property_tree::ptree ptree_config;
    boost::property_tree::read_xml(m_config_file, ptree_config, boost::property_tree::xml_parser::trim_whitespace);

    //
    // Create subconfiguations.
    //

    std::cout << "creating subconfigurations..." << std::endl;
    std::stringstream datasets_info;
    boost::property_tree::write_xml(datasets_info,ptree_config.get_child("gbkfit.datasets"));
    std::stringstream instrument_info;
    boost::property_tree::write_xml(instrument_info, ptree_config.get_child("gbkfit.instrument"));
    std::stringstream model_config_info;
    boost::property_tree::write_xml(model_config_info, ptree_config.get_child("gbkfit.model_config"));
    std::stringstream fitter_config_info;
    boost::property_tree::write_xml(fitter_config_info, ptree_config.get_child("gbkfit.fitter_config"));
    std::stringstream params_config_info;
    boost::property_tree::write_xml(params_config_info, ptree_config.get_child("gbkfit.params_config"));

    //
    // Create the components.
    //

    std::cout << "setting up datasets..." << std::endl;
    m_datasets = m_core->create_datasets(datasets_info.str());

    std::cout << "setting up instrument..." << std::endl;
    m_instrument = m_core->create_instrument(instrument_info.str());

    std::cout << "setting up model..." << std::endl;
    m_model = m_core->create_model(model_config_info.str());

    std::cout << "setting up fitter..." << std::endl;
    m_fitter = m_core->create_fitter(fitter_config_info.str());

    std::cout << "setting up parameters..." << std::endl;
    m_parameters = m_core->create_parameters(params_config_info.str());

    //
    // All done!
    //

    std::cout << "initialization completed" << std::endl;

    return true;
}

void Application::shutdown(void)
{
    std::cout << "shutdown started" << std::endl;

    delete m_core;

    for(std::size_t i = 0; i < m_model_factories.size(); ++i) {
        delete m_model_factories[i];
    }

    for(std::size_t i = 0; i < m_fitter_factories.size(); ++i) {
        delete m_fitter_factories[i];
    }

    delete m_model;
    delete m_fitter;
    delete m_parameters;
    delete m_instrument;

    std::cout << "shutdown completed" << std::endl;
}

void Application::run(void)
{
    std::cout << "main execution path started" << std::endl;

    // Initialize model.
    int model_size_x = m_datasets.begin()->second->get_data()->get_shape()[0];
    int model_size_y = m_datasets.begin()->second->get_data()->get_shape()[1];
    int model_size_z = 161;
    m_model->initialize(model_size_x,model_size_y,model_size_z,m_instrument);

    //
    // Fit!
    //

    std::cout << "fitting started" << std::endl;
    m_fitter->fit(m_model,m_datasets,*m_parameters);
    std::cout << "fitting complete" << std::endl;

    //
    // Evaluate model with best fit parameters.
    //

    std::map<std::string, float> params;
    for(auto& param_name : m_model->get_parameter_names()) {
        params.emplace(param_name, m_parameters->get_parameter(param_name).get<float>("best"));
    }

    std::map<std::string, gbkfit::NDArray*> model_data = m_model->evaluate(params);

    float* velmap_mdl = model_data["velmap"]->map();
    float* sigmap_mdl = model_data["sigmap"]->map();

    float* velmap_data_d = m_datasets["velmap"]->get_data()->map();
    float* velmap_data_e = m_datasets["velmap"]->get_errors()->map();
    float* velmap_data_m = m_datasets["velmap"]->get_mask()->map();

    float* sigmap_data_d = m_datasets["sigmap"]->get_data()->map();
    float* sigmap_data_e = m_datasets["sigmap"]->get_errors()->map();
    float* sigmap_data_m = m_datasets["sigmap"]->get_mask()->map();


    int dof = 0;
    float chi2 = 0;
    float chi2_red = 0;
    for(int i = 0; i < model_size_x*model_size_y; ++i)
    {
        float vel_m = velmap_data_m[i];
        float sig_m = sigmap_data_m[i];

        if (vel_m > 0) {
            float residual = (velmap_data_d[i] - velmap_mdl[i]) / velmap_data_e[i];
            chi2 += (residual*residual);
            dof++;
        }

        if (sig_m > 0) {
            float residual = (sigmap_data_d[i] - sigmap_mdl[i]) / sigmap_data_e[i];
            chi2 += (residual*residual);
            dof++;
        }
    }

    chi2_red = chi2 / dof;

    std::cout << "chi2: " << chi2 << std::endl
              << "chi2_red: " << chi2_red << std::endl;
    /*
    model_data["velmap"]->unmap();
    model_data["sigmap"]->unmap();
    m_datasets["velmap"]->get_data()->unmap();
    m_datasets["sigmap"]->get_data()->unmap();
    */

    //
    // Output results
    //

    boost::property_tree::ptree ptree_results;

    std::vector<std::string> param_options = { "name", "best", "mean", "stddev", "map" };

#if 0
    for(auto& param_name : m_model->get_parameter_names()) {
        boost::property_tree::ptree ptree_parameter;
        for(auto& option_name : param_options) {
            if (m_parameters->get_parameter(param_name).has(option_name)) {
                ptree_parameter.add("<xmlattr>." + option_name, m_parameters->get_parameter(param_name).get<std::string>(option_name));
            }
        }
        ptree_results.add_child("gbkfit.results.parameters.parameter", ptree_parameter);
    }
    boost::property_tree::xml_writer_settings<std::string> settings(' ', 2);
    boost::property_tree::write_xml("gbkfit_results.xml", ptree_results, std::locale(), settings);

#else

    boost::property_tree::ptree ptree_array;
    for(auto& param_name : m_model->get_parameter_names())
    {
        boost::property_tree::ptree ptree_parameter;
        for(auto& option_name : param_options)
        {
            if (m_parameters->get_parameter(param_name).has(option_name))
            {
                ptree_parameter.put(option_name, m_parameters->get_parameter(param_name).get<std::string>(option_name));
            }

        }
        ptree_array.push_back(std::make_pair("", ptree_parameter));
    }

    ptree_results.add_child("parameters", ptree_array);

    boost::property_tree::write_json("gbkfit_results.json", ptree_results, std::locale(), true);

#endif

    // Write data products to disk.
    #if 1
    gbkfit::fits::write_to("!flxcube_up.fits",*model_data["flxcube_up"]);
    gbkfit::fits::write_to("!psfcube_up.fits",*model_data["psfcube_up"]);
    gbkfit::fits::write_to("!psfcube_u.fits",*model_data["psfcube_u"]);
    gbkfit::fits::write_to("!psfcube.fits",*model_data["psfcube"]);
    gbkfit::fits::write_to("!flxcube_mdl.fits",*model_data["flxcube"]);
    gbkfit::fits::write_to("!flxmap_mdl.fits",*model_data["flxmap"]);
    gbkfit::fits::write_to("!velmap_mdl.fits",*model_data["velmap"]);
    gbkfit::fits::write_to("!sigmap_mdl.fits",*model_data["sigmap"]);

    /*
    gbkfit::fits::write_to("!" + m_galaxy_id + "_velmap_res.fits",*resid_data_velmap);
    gbkfit::fits::write_to("!" + m_galaxy_id + "_sigmap_res.fits",*resid_data_sigmap);
    gbkfit::fits::write_to("!" + m_galaxy_id + "_velmap_res_w.fits",*resid_data_velmap_err);
    gbkfit::fits::write_to("!" + m_galaxy_id + "_sigmap_res_w.fits",*resid_data_sigmap_err);
    */
    #endif

    //
    //  Done!
    //

    std::cout << "main execution path finished" << std::endl;
}

} // namespace gbkfit_app_cli
