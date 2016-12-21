
#include "application.hpp"

#include <boost/program_options.hpp>

#include "gbkfit/core.hpp"
#include "gbkfit/dataset.hpp"
#include "gbkfit/dmodel.hpp"
#include "gbkfit/fitter.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/fitter_result.hpp"
#include "gbkfit/gmodel.hpp"
#include "gbkfit/json.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/params.hpp"
#include "gbkfit/utility.hpp"
#include "gbkfit/version.hpp"

#ifdef GBKFIT_BUILD_DMODEL_OMP
#include "gbkfit/dmodel/mmaps/mmaps_omp.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_omp_factory.hpp"
#include "gbkfit/dmodel/scube/scube_omp.hpp"
#include "gbkfit/dmodel/scube/scube_omp_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_DMODEL_CUDA
#include "gbkfit/dmodel/mmaps/mmaps_cuda.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_cuda_factory.hpp"
#include "gbkfit/dmodel/scube/scube_cuda.hpp"
#include "gbkfit/dmodel/scube/scube_cuda_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_GMODEL_OMP
#include "gbkfit/gmodel/gmodel1/gmodel1_omp.hpp"
#include "gbkfit/gmodel/gmodel1/gmodel1_omp_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_GMODEL_CUDA
#include "gbkfit/gmodel/gmodel1/gmodel1_cuda.hpp"
#include "gbkfit/gmodel/gmodel1/gmodel1_cuda_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_FITTER_MPFIT
#include "gbkfit/fitter/mpfit/fitter_mpfit.hpp"
#include "gbkfit/fitter/mpfit/fitter_mpfit_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_FITTER_MULTINEST
#include "gbkfit/fitter/multinest/fitter_multinest.hpp"
#include "gbkfit/fitter/multinest/fitter_multinest_factory.hpp"
#endif

namespace gbkfit_app_cli {

const std::string Application::DEFAULT_CONFIG_FILE = "../../data/configs/gbkfit_config_fit_2d.json";
//const std::string Application::DEFAULT_CONFIG_FILE = "config_fit_3d.json";
//const std::string Application::DEFAULT_CONFIG_FILE = "gbkfit_config.json";
const std::string Application::DEFAULT_OUTPUT_DIR = "output";

Application::Application(void)
    : m_config_file(DEFAULT_CONFIG_FILE)
    , m_output_dir(DEFAULT_OUTPUT_DIR)
    , m_core(nullptr)
    , m_dmodel(nullptr)
    , m_gmodel(nullptr)
    , m_fitter(nullptr)
    , m_params(nullptr)
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
            ("output", boost::program_options::value<std::string>(&m_output_dir),  "")
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

bool Application::initialize(void)
{
    std::cout << "GBKFIT version: " << gbkfit::VERSION << std::endl;

    std::cout << "initialization started" << std::endl;

    //
    // Create gbkfit core.
    //

    std::cout << "creating gbkfit core..." << std::endl;
    m_core = new gbkfit::Core();

    //
    // Create factories.
    //

    std::cout << "creating dmodel factories..." << std::endl;
    #ifdef GBKFIT_BUILD_DMODEL_OMP
    m_dmodel_factories.push_back(new gbkfit::dmodel::mmaps::MMapsOmpFactory());
    m_dmodel_factories.push_back(new gbkfit::dmodel::scube::SCubeOmpFactory());
    #endif
    #ifdef GBKFIT_BUILD_DMODEL_CUDA
    m_dmodel_factories.push_back(new gbkfit::dmodel::mmaps::MMapsCudaFactory());
    m_dmodel_factories.push_back(new gbkfit::dmodel::scube::SCubeCudaFactory());
    #endif

    std::cout << "creating gmodel factories..." << std::endl;
    #ifdef GBKFIT_BUILD_GMODEL_OMP
    m_gmodel_factories.push_back(new gbkfit::gmodel::gmodel1::GModel1OmpFactory());
    #endif
    #ifdef GBKFIT_BUILD_GMODEL_CUDA
    m_gmodel_factories.push_back(new gbkfit::gmodel::gmodel1::GModel1CudaFactory());
    #endif

    std::cout << "creating fitter factories..." << std::endl;
    #ifdef GBKFIT_BUILD_FITTER_MPFIT
    m_fitter_factories.push_back(new gbkfit::fitter::mpfit::FitterMpfitFactory());
    #endif
    #ifdef GBKFIT_BUILD_FITTER_MULTINEST
    m_fitter_factories.push_back(new gbkfit::fitter::multinest::FitterMultinestFactory());
    #endif

    //
    // Register factories.
    //

    std::cout << "registering dmodel factories..." << std::endl;
    for(std::size_t i = 0; i < m_dmodel_factories.size(); ++i) {
        std::cout << "registering dmodel factory '" << m_dmodel_factories[i]->get_type() << "'..." << std::endl;
        m_core->add_dmodel_factory(m_dmodel_factories[i]);
    }

    std::cout << "registering gmodel factories..." << std::endl;
    for(std::size_t i = 0; i < m_gmodel_factories.size(); ++i) {
        std::cout << "registering gmodel factory '" << m_gmodel_factories[i]->get_type() << "'..." << std::endl;
        m_core->add_gmodel_factory(m_gmodel_factories[i]);
    }

    std::cout << "registering fitter factories..." << std::endl;
    for(std::size_t i = 0; i < m_fitter_factories.size(); ++i) {
        std::cout << "registering fitter factory '" << m_fitter_factories[i]->get_type() << "'..." << std::endl;
        m_core->add_fitter_factory(m_fitter_factories[i]);
    }

    //
    // Read configuration.
    //

    std::cout << "reading configuration from '" << m_config_file << "'..." << std::endl;
    std::ifstream config_stream(m_config_file);
    nlohmann::json config = nlohmann::json::parse(config_stream);

    //
    // Patch-up configuration
    //

    // Datasets section should always be there, even if it is empty.
    if (config.count("datasets") == 0) {
        config["datasets"] = nlohmann::json::array();
    }

    //
    // Read mode.
    //

    std::cout << "reading mode..." << std::endl;
    m_task = config.at("mode");

    if (m_task != "evaluate" && m_task != "fit") {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    //
    // Create the components.
    //

    std::cout << "setting up datasets..." << std::endl;
    m_datasets = m_core->create_datasets(config.at("datasets").dump());

    std::cout << "setting up point spread function..." << std::endl;
    m_psf = m_core->create_point_spread_function(config.at("psf").dump());

    std::cout << "setting up line spread function..." << std::endl;
    m_lsf = m_core->create_line_spread_function(config.at("lsf").dump());

    std::cout << "setting up gmodel..." << std::endl;
    m_gmodel = m_core->create_gmodel(config.at("gmodel").dump());

    std::cout << "setting up params..." << std::endl;
    m_params = m_core->create_parameters(config.at("params").dump());


    std::vector<int> shape;
    if (m_task == "fit")
    {
        std::cout << "setting up fitter..." << std::endl;
        m_fitter = m_core->create_fitter(config.at("fitter").dump());

        if (m_datasets.empty()) {
            throw std::runtime_error(BOOST_CURRENT_FUNCTION);
        }

        shape = m_datasets[0]->get_data()->get_shape().as_vector();
    }

    std::cout << "setting up dmodel..." << std::endl;
    m_dmodel = m_core->create_dmodel(config.at("dmodel").dump(), shape, {}, m_psf, m_lsf);

    m_dmodel->set_galaxy_model(m_gmodel);

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

    for(auto& factory : m_fitter_factories)
        delete factory;

    for(auto& factory : m_dmodel_factories)
        delete factory;

    for(auto& factory : m_gmodel_factories)
        delete factory;

    std::cout << "shutdown completed" << std::endl;
}

void Application::run(void)
{
    std::cout << "main execution path started" << std::endl;

    /*
    std::chrono::time_point<std::chrono::system_clock> t1;
    std::chrono::time_point<std::chrono::system_clock> t2;
    t1 = std::chrono::system_clock::now();
    t2 = std::chrono::system_clock::now();
    std::chrono::duration_cas<std::chrono::milliseconds>(t2-t1);
    */

    if (m_task == "fit")
    {
        gbkfit::FitterResult* result = m_fitter->fit(m_dmodel, m_params, m_datasets);
        std::cout << result->to_string() << std::endl;
        result->save("results.json");
    }
    else // (m_task == "evaluate")
    {
        std::map<std::string, float> params = m_params->get_map<float>("value");
        std::map<std::string, gbkfit::NDArrayHost*> model_data = m_dmodel->evaluate(params);
        for(auto& data : model_data){
            gbkfit::fits::write_to("!"+data.first+"_model.fits", *data.second);
        }
    }

    std::cout << "main execution path completed" << std::endl;
}

} // namespace gbkfit_app_cli
