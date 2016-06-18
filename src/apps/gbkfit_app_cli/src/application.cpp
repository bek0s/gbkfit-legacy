
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

#ifdef GBKFIT_BUILD_DMODEL_MMAPS_CUDA
#include "gbkfit/dmodel/mmaps/mmaps_cuda.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_cuda_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_DMODEL_MMAPS_OMP
#include "gbkfit/dmodel/mmaps/mmaps_omp.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_omp_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_DMODEL_SCUBE_CUDA
#include "gbkfit/dmodel/scube/scube_cuda.hpp"
#include "gbkfit/dmodel/scube/scube_cuda_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_GMODEL_GMODEL1_CUDA
#include "gbkfit/gmodel/gmodel1/gmodel1_cuda.hpp"
#include "gbkfit/gmodel/gmodel1/gmodel1_cuda_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_GMODEL_GMODEL1_OMP
#include "gbkfit/gmodel/gmodel1/gmodel1_omp.hpp"
#include "gbkfit/gmodel/gmodel1/gmodel1_omp_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_DMODEL_SCUBE_OMP
#include "gbkfit/dmodel/scube/scube_omp.hpp"
#include "gbkfit/dmodel/scube/scube_omp_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_FITTER_MPFIT
#include "gbkfit/fitter/mpfit/fitter_mpfit.hpp"
#include "gbkfit/fitter/mpfit/fitter_mpfit_factory.hpp"
#endif

#ifdef GBKFIT_BUILD_FITTER_MULTINEST
#include "gbkfit/fitter/multinest/fitter_multinest.hpp"
#include "gbkfit/fitter/multinest/fitter_multinest_factory.hpp"
#endif

#include "gbkfit/instrument.hpp"
#include "gbkfit/spread_functions.hpp"

namespace gbkfit_app_cli {

const std::string Application::DEFAULT_CONFIG_FILE = "../../data/configs/gbkfit_config_2d.json";
const std::string Application::DEFAULT_OUTPUT_DIR = "output";

Application::Application(void)
    : m_config_file(DEFAULT_CONFIG_FILE)
    , m_output_dir(DEFAULT_OUTPUT_DIR)
    , m_core(nullptr)
    , m_dmodel(nullptr)
    , m_gmodel(nullptr)
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
    std::cout << "initialization started" << std::endl;

    //
    // Create gbkfit core.
    //

    m_core = new gbkfit::Core();

    //
    // Create factories.
    //

    std::cout << "creating dmodel factories..." << std::endl;
    #ifdef GBKFIT_BUILD_DMODEL_MMAPS_CUDA
    m_dmodel_factories.push_back(new gbkfit::dmodel::mmaps::MMapsCudaFactory());
    #endif
    #ifdef GBKFIT_BUILD_DMODEL_MMAPS_OMP
    m_dmodel_factories.push_back(new gbkfit::dmodel::mmaps::MMapsOmpFactory());
    #endif
    #ifdef GBKFIT_BUILD_DMODEL_SCUBE_CUDA
    m_dmodel_factories.push_back(new gbkfit::dmodel::scube::SCubeCudaFactory());
    #endif
    #ifdef GBKFIT_BUILD_DMODEL_SCUBE_OMP
    m_dmodel_factories.push_back(new gbkfit::dmodel::scube::SCubeOmpFactory());
    #endif

    std::cout << "creating gmodel factories..." << std::endl;
    #ifdef GBKFIT_BUILD_GMODEL_GMODEL1_CUDA
    m_gmodel_factories.push_back(new gbkfit::gmodel::gmodel1::GModel1CudaFactory());
    #endif
    #ifdef GBKFIT_BUILD_GMODEL_GMODEL1_OMP
    m_gmodel_factories.push_back(new gbkfit::gmodel::gmodel1::GModel1OmpFactory());
    #endif

    std::cout << "creating fitter factories..." << std::endl;
    #ifdef GBKFIT_BUILD_FITTER_MPFIT
    m_fitter_factories.push_back(new gbkfit::fitter::mpfit::FitterMpfitFactory());
    #endif
    #ifdef GBKFIT_BUILD_FITTER_MULTINEST
    m_fitter_factories.push_back(new gbkfit::fitter::multinest::FitterMultinestFactory());
    #endif

    //
    //  Add factories to the gbkfit core.
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

    std::cout << "reading configuration..." << std::endl;

    std::ifstream config_stream(m_config_file);
    nlohmann::json config_root = nlohmann::json::parse(config_stream);

//  std::cout << "reading task type..." << std::endl;

//  m_task = config_root.at("task");

    //
    // Create subconfiguations.
    //

    std::cout << "creating subconfigurations..." << std::endl;

    std::string config_datasets   = config_root.at("datasets").dump();
    std::string config_instrument = config_root.at("instrument").dump();

    std::string config_dmodel = config_root.at("dmodel").dump();
    std::string config_gmodel = config_root.at("gmodel").dump();
    std::string config_fitter = config_root.at("fitter").dump();
    std::string config_parameters = config_root.at("parameters").dump();

    //
    // Create the components.
    //

    std::cout << "setting up datasets..." << std::endl;
    m_datasets = m_core->create_datasets(config_datasets);

    std::cout << "setting up instrument..." << std::endl;
    m_instrument = m_core->create_instrument(config_instrument);

    std::cout << "setting up fitter..." << std::endl;
    m_fitter = m_core->create_fitter(config_fitter);

    std::cout << "setting up parameters..." << std::endl;
    m_parameters = m_core->create_parameters(config_parameters);

    std::cout << "setting up dmodel..." << std::endl;
    std::vector<int> shape = m_datasets[0]->get_data()->get_shape().as_vector();
    m_dmodel = m_core->create_dmodel(config_dmodel, shape, m_instrument);

    std::cout << "setting up gmodel..." << std::endl;
    m_gmodel = m_core->create_gmodel(config_gmodel);

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

void test_models(int hardware_mode, const std::string& output_prefix)
{
    // Instrument configuration
    float psf_fwhm = 2.5f;
    float lsf_fwhm = 30.0f;
    float sampling_x = 1.0f;
    float sampling_y = 1.0f;
    float sampling_z = 10.0f;

    // Data model configuration
    int size_x = 49;
    int size_y = 49;
    int size_z = 81;

    // Galaxy model configuration
    std::map<std::string, float> params = {
        {"i0", 1},
        {"r0", 10},
        {"xo", 24.5},
        {"yo", 24.5},
        {"pa", 45},
        {"incl", 70},
        {"rt", 4},
        {"vt", 200},
        {"vsys", 0},
        {"vsig", 50},
        {"a", 1},
        {"b", 1}
    };

    //
    // Create instrument
    //

    gbkfit::PointSpreadFunction* psf = nullptr;
    gbkfit::LineSpreadFunction* lsf = nullptr;
//  psf = new gbkfit::PointSpreadFunctionNone();
//  lsf = new gbkfit::LineSpreadFunctionNone();
    psf = new gbkfit::PointSpreadFunctionGaussian(psf_fwhm);
    lsf = new gbkfit::LineSpreadFunctionGaussian(lsf_fwhm);

    gbkfit::Instrument* instrument = new gbkfit::Instrument(sampling_x,
                                                            sampling_y,
                                                            sampling_z,
                                                            psf,
                                                            lsf);

    //
    // Create data and galaxy models
    //

    gbkfit::GModel* gmodel = nullptr;
    gbkfit::DModel* dmodel_scube = nullptr;
    gbkfit::DModel* dmodel_mmaps = nullptr;

    if      (hardware_mode == 0)
    {
        gmodel = new gbkfit::gmodel::gmodel1::GModel1Omp(
                gbkfit::gmodel::gmodel1::FlxProfileType::exponential,
                gbkfit::gmodel::gmodel1::VelProfileType::arctan);

        dmodel_scube = new gbkfit::dmodel::scube::SCubeOmp(
                size_x, size_y, size_z, 1, 1, 1, instrument);

        dmodel_mmaps = new gbkfit::dmodel::mmaps::MMapsOmp(
                size_x, size_y, 1, 1, instrument);
    }
    else if (hardware_mode == 1)
    {
        gmodel = new gbkfit::gmodel::gmodel1::GModel1Cuda(
                gbkfit::gmodel::gmodel1::FlxProfileType::exponential,
                gbkfit::gmodel::gmodel1::VelProfileType::arctan);

        dmodel_scube = new gbkfit::dmodel::scube::SCubeCuda(
                size_x, size_y, size_z, 1, 1, 1, instrument);

        dmodel_mmaps = new gbkfit::dmodel::mmaps::MMapsCuda(
                size_x, size_y, 1, 1, instrument);
    }

    dmodel_scube->set_galaxy_model(gmodel);
    dmodel_mmaps->set_galaxy_model(gmodel);

    //
    // Evaluate models and dump results to the filesystem
    //

    std::map<std::string, gbkfit::NDArrayHost*> scube_data = dmodel_scube->evaluate(params);
    gbkfit::fits::write_to("!"+output_prefix+"_flxcube.fits", *scube_data.at("flxcube"));

    std::map<std::string, gbkfit::NDArrayHost*> mmaps_data = dmodel_mmaps->evaluate(params);
    gbkfit::fits::write_to("!"+output_prefix+"_flxmap.fits", *mmaps_data.at("flxmap"));
    gbkfit::fits::write_to("!"+output_prefix+"_sigmap.fits", *mmaps_data.at("sigmap"));
    gbkfit::fits::write_to("!"+output_prefix+"_velmap.fits", *mmaps_data.at("velmap"));

    delete gmodel;
    delete dmodel_scube;
    delete dmodel_mmaps;
    delete instrument;
    delete psf;
    delete lsf;
}

void Application::run(void)
{

#if 0
    test_models(0, "omp");
    test_models(1, "cuda");
    return;
#endif

    std::cout << "main execution path started" << std::endl;

    //
    // Fit!
    //

    std::cout << "fitting started" << std::endl;
    gbkfit::FitterResult* result = m_fitter->fit(m_dmodel, m_parameters, m_datasets);
    std::cout << "fitting complete" << std::endl;


    std::cout << result->to_string() << std::endl;
    result->save("results.json");

    //
    //  Done!
    //

    std::cout << "main execution path finished" << std::endl;
}

} // namespace gbkfit_app_cli
