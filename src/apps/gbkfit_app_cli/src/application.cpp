
#include "application.hpp"

#include "gbkfit/core.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/nddataset.hpp"

#include "gbkfit/fitters/mpfit/fitter_mpfit.hpp"
#include "gbkfit/fitters/multinest/fitter_multinest.hpp"
#include "gbkfit/models/model01/model_model01.hpp"
#include "gbkfit/models/model01/model_model01_omp.hpp"
#include "gbkfit/models/model01/model_model01_cuda.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "gbkfit/utility.hpp"
#include "gbkfit/parameters_fit_info.hpp"


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
    std::cout << "Initialization started." << std::endl;

    // create gbkfit core
    m_core = new gbkfit::core();

    // create model factories
    m_model_factory_model01_cuda = new gbkfit::models::model01::model_factory_model01_cuda();
    m_model_factory_model01_omp = new gbkfit::models::model01::model_factory_model01_omp();

    // create fitter factories
    m_fitter_factory_mpfit = new gbkfit::fitters::mpfit::fitter_factory_mpfit();
    m_fitter_factory_multinest = new gbkfit::fitters::multinest::fitter_factory_multinest();

    // add model factories to the gbkfit core
    m_core->add_model_factory(m_model_factory_model01_cuda);
    m_core->add_model_factory(m_model_factory_model01_omp);

    // add fitter factories to the gbkfit core
    m_core->add_fitter_factory(m_fitter_factory_mpfit);
    m_core->add_fitter_factory(m_fitter_factory_multinest);

    // read configuration
    boost::property_tree::ptree ptree_config;
    boost::property_tree::read_xml("../../config/test_config_01.xml",ptree_config, boost::property_tree::xml_parser::trim_whitespace);

    // get model info
    std::stringstream model_info;
    boost::property_tree::write_xml(model_info,ptree_config.get_child("gbkfit.model"));

    // get fitter info
    std::stringstream fitter_info;
    boost::property_tree::write_xml(fitter_info,ptree_config.get_child("gbkfit.fitter"));

    // ...
    std::stringstream parameters_info;
    boost::property_tree::write_xml(parameters_info,ptree_config.get_child("gbkfit.parameters"));

    // get detasets info
    std::stringstream datasets_info;
    boost::property_tree::write_xml(datasets_info,ptree_config.get_child("gbkfit.datasets"));

    std::stringstream instrument_info;
    boost::property_tree::write_xml(instrument_info,ptree_config.get_child("gbkfit.instrument"));

    // create datasets
    m_datasets = m_core->create_datasets(datasets_info.str());

    // create model
    m_model = m_core->create_model(model_info.str());

    // create fitter
    m_fitter = m_core->create_fitter(fitter_info.str());

    m_fit_info = m_core->create_parameters_fit_info(parameters_info.str());

    m_instrument = m_core->create_instrument(instrument_info.str());

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
        dataset.second->__destroy();
        delete dataset.second;
    }

    std::cout << "Shutdown completed." << std::endl;
}

void application::run(void)
{
    std::cout << "Main loop started." << std::endl;

    if(m_model)
    {
        int model_size = 33;
        int model_size_x = model_size;
        int model_size_y = model_size;
        float xo = model_size_x/2.0;
        float yo = model_size_y/2.0;
        std::cout << "xo=" << xo << ", "
                  << "yo=" << yo << std::endl;

        m_model->initialize(model_size,model_size,100,m_instrument);


        std::map<std::string,float> params = {
            {"vsys",0},
            {"xo",xo},
            {"yo",yo},
            {"pa",90},
            {"incl",45},
            {"i0",1.0},
            {"r0",2.0},
            {"rt",5.0},
            {"vt",200},
            {"vsig",30}
        };

        std::map<std::string,gbkfit::ndarray*> data = m_model->evaluate(params);
    //  gbkfit::fits::write_to("!foo_flxcube_padded.fits",*data["flxcube_padded"]);
    //  gbkfit::fits::write_to("!foo_psfcube_padded.fits",*data["psfcube_padded"]);
        gbkfit::fits::write_to("!foo_flxcube.fits",*data["flxcube"]);
        gbkfit::fits::write_to("!foo_flxmap.fits",*data["flxmap"]);
        gbkfit::fits::write_to("!foo_velmap.fits",*data["velmap"]);
        gbkfit::fits::write_to("!foo_sigmap.fits",*data["sigmap"]);
    }

//  m_fitter->fit(m_model,m_datasets,*m_fit_info);

    std::cout << "Main loop finished." << std::endl;
}


} // namespace gbkfit_app_cli
