
#include "application.hpp"

#include "gbkfit/core.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/nddataset.hpp"

#include "gbkfit/fitters/mpfit/fitter_mpfit.hpp"
//#include "gbkfit/fitters/multinest/fitter_multinest.hpp"
#include "gbkfit/models/model01/model_model01.hpp"
#include "gbkfit/models/model01/model_model01_omp.hpp"
//#include "gbkfit/models/model01/model_model01_cuda.hpp"

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "gbkfit/utility.hpp"
#include "gbkfit/parameters_fit_info.hpp"

#include "gbkfit/instrument.hpp"


namespace gbkfit_app_cli {


application::application(void)
    : m_core(nullptr)
    , m_model_factory_model01_cuda(nullptr)
    , m_model_factory_model01_omp(nullptr)
    , m_fitter_factory_mpfit(nullptr)
    , m_fitter_factory_multinest(nullptr)
    , m_model(nullptr)
    , m_fitter(nullptr)
    , m_fit_info(nullptr)
    , m_instrument(nullptr)
{
}

application::~application()
{
}

bool application::initialize(void)
{
    /*
    boost::program_options::options_description po_desc("options");
    po_desc.add_options()
            ("config", boost::program_options::value<std::string>(&m_filename_config), "")
            ("help",                                                                   "");

    boost::program_options::variables_map po_vm;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc,argv,po_desc),po_vm);
        boost::program_options::notify(po_vm);
    } catch(boost::program_options::error& e) {
        std::cout << "Could not parse command-line arguments. Error: " << e.what() << std::endl;
        std::cout << po_desc;
        return false;
    }

    if(po_vm.count("help")) {
        std::cout << po_desc;
        return false;
    }
    */

    std::cout << "Initialization started." << std::endl;

    //
    // Create gbkfit core.
    //

    m_core = new gbkfit::core();

    //
    // Create factories and add the to the gbkfit core.
    //

//  m_model_factory_model01_cuda = new gbkfit::models::model01::model_factory_model01_cuda();
    m_model_factory_model01_omp = new gbkfit::models::model01::model_factory_model01_omp();

    m_fitter_factory_mpfit = new gbkfit::fitters::mpfit::fitter_factory_mpfit();
//  m_fitter_factory_multinest = new gbkfit::fitters::multinest::fitter_factory_multinest();

//  m_core->add_model_factory(m_model_factory_model01_cuda);
    m_core->add_model_factory(m_model_factory_model01_omp);

    m_core->add_fitter_factory(m_fitter_factory_mpfit);
//  m_core->add_fitter_factory(m_fitter_factory_multinest);

    //
    // Read XML configuration and prepare to send it to different components.
    //

    boost::property_tree::ptree ptree_config;
    boost::property_tree::read_xml("../../config/test_config_01.xml",ptree_config, boost::property_tree::xml_parser::trim_whitespace);

    std::stringstream model_info;
    boost::property_tree::write_xml(model_info,ptree_config.get_child("gbkfit.model"));

    std::stringstream fitter_info;
    boost::property_tree::write_xml(fitter_info,ptree_config.get_child("gbkfit.fitter"));

    std::stringstream parameters_info;
    boost::property_tree::write_xml(parameters_info,ptree_config.get_child("gbkfit.parameters"));

    std::stringstream datasets_info;
    boost::property_tree::write_xml(datasets_info,ptree_config.get_child("gbkfit.datasets"));

    std::stringstream instrument_info;
    boost::property_tree::write_xml(instrument_info,ptree_config.get_child("gbkfit.instrument"));

    //
    // Create the appropriate components based on the configuration read.
    //

    m_datasets = m_core->create_datasets(datasets_info.str());

    m_model = m_core->create_model(model_info.str());

    m_fitter = m_core->create_fitter(fitter_info.str());

    m_fit_info = m_core->create_parameters_fit_info(parameters_info.str());

    m_instrument = m_core->create_instrument(instrument_info.str());

    //
    // All done!
    //

    std::cout << "Initialization completed." << std::endl;

    return true;
}

void application::shutdown(void)
{
    std::cout << "Shutdown started." << std::endl;

    delete m_core;
    delete m_model_factory_model01_cuda;
    delete m_model_factory_model01_omp;
    delete m_fitter_factory_mpfit;
    delete m_fitter_factory_multinest;
    delete m_model;
    delete m_fitter;
    delete m_fit_info;
    delete m_instrument;

    for(auto& dataset : m_datasets)
    {
        dataset.second->__destroy(); // uughh.. ugly, I know, it is temporary.
        delete dataset.second;
    }

    std::cout << "Shutdown completed." << std::endl;
}

void application::run(void)
{
    std::cout << "Main loop started." << std::endl;

    // Initialize model.
//  int model_size_x = 49; // m_datasets.begin()->second->get_data("data")->get_shape()[0];
//  int model_size_y = 49; // m_datasets.begin()->second->get_data("data")->get_shape()[1];
    int model_size_x = m_datasets.begin()->second->get_data("data")->get_shape()[0];
    int model_size_y = m_datasets.begin()->second->get_data("data")->get_shape()[1];
    int model_size_z = 101;
    m_model->initialize(model_size_x,model_size_y,model_size_z,m_instrument);

    /*
    float xo = model_size_x/2.0;
    float yo = model_size_y/2.0;

    std::map<std::string,float> params = {
        {"vsig",50},
        {"vsys",0},
        {"xo",xo},
        {"yo",yo},
        {"pa",45},
        {"incl",45},
        {"i0",1.0},
        {"r0",10.0},
        {"rt",4.0},
        {"vt",200}
    };

    std::map<std::string,gbkfit::ndarray*> data = m_model->evaluate(params);
    gbkfit::fits::write_to("!foo_flxcube_up.fits",*data["flxcube_up"]);
    gbkfit::fits::write_to("!foo_psfcube_up.fits",*data["psfcube_up"]);
    gbkfit::fits::write_to("!foo_psfcube_u.fits",*data["psfcube_u"]);
    gbkfit::fits::write_to("!foo_psfcube.fits",*data["psfcube"]);
    gbkfit::fits::write_to("!foo_flxcube.fits",*data["flxcube"]);
    gbkfit::fits::write_to("!foo_flxmap.fits",*data["flxmap"]);
    gbkfit::fits::write_to("!foo_velmap.fits",*data["velmap"]);
    gbkfit::fits::write_to("!foo_sigmap.fits",*data["sigmap"]);
    */

    // Fit (at last!)
    m_fitter->fit(m_model,m_datasets,*m_fit_info);

    std::cout << "Main loop finished." << std::endl;
}


} // namespace gbkfit_app_cli
