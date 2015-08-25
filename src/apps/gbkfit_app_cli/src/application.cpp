
#include "application.hpp"

#include "gbkfit/core.hpp"
#include "gbkfit/fits_util.hpp"
#include "gbkfit/fitters/mpfit/fitter_mpfit.hpp"
#include "gbkfit/fitters/multinest/fitter_multinest.hpp"
#include "gbkfit/models/galaxy_2d/model_galaxy_2d.hpp"
#include "gbkfit/models/galaxy_2d_omp/model_galaxy_2d_omp.hpp"
#include "gbkfit/models/galaxy_2d_cuda/model_galaxy_2d_cuda.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit_app_cli {


application::application(void)
    : m_core(nullptr)
    , m_fitter_factory_mpfit(nullptr)
    , m_fitter_factory_multinest(nullptr)
    , m_model_factory_galaxy_2d_cuda(nullptr)
    , m_model_factory_galaxy_2d_omp(nullptr)
    , m_fitter(nullptr)
    , m_model(nullptr)
{
}

application::~application()
{
}


bool application::initialize(void)
{
    std::cout << "initialization started" << std::endl;

    // create gbkfit core
    m_core = new gbkfit::core();

    // create model factories
    m_model_factory_galaxy_2d_cuda = new gbkfit::models::galaxy_2d::model_factory_galaxy_2d_cuda();
    m_model_factory_galaxy_2d_omp = new gbkfit::models::galaxy_2d::model_factory_galaxy_2d_omp();

    // create fitter factories
    m_fitter_factory_mpfit = new gbkfit::fitters::mpfit::fitter_factory_mpfit();
    m_fitter_factory_multinest = new gbkfit::fitters::multinest::fitter_factory_multinest();

    // add model factories to the gbkfit core
    m_core->add_model_factory(m_model_factory_galaxy_2d_cuda);
    m_core->add_model_factory(m_model_factory_galaxy_2d_omp);

    // add fitter factories to the gbkfit core
    m_core->add_fitter_factory(m_fitter_factory_mpfit);
    m_core->add_fitter_factory(m_fitter_factory_multinest);

    // read configuration
    boost::property_tree::ptree ptree_config;
    boost::property_tree::read_xml("../../tools/gbkfit_config.xml",ptree_config);

    // get model info
    std::stringstream model_info;
    boost::property_tree::write_xml(model_info,ptree_config.get_child("gbkfit.config.model"));

    // get fitter info
    std::stringstream fitter_info;
    boost::property_tree::write_xml(fitter_info,ptree_config.get_child("gbkfit.config.fitter"));

    // get detasets info
    std::stringstream datasets_info;
    boost::property_tree::write_xml(datasets_info,ptree_config.get_child("gbkfit.config.datasets"));

    // create datasets
//  m_datasets = m_core->create_datasets(datasets_info);

    // create model
    m_model = m_core->create_model(model_info);

    // create fitter
    m_fitter = m_core->create_fitter(fitter_info);

    std::cout << "initialization completed" << std::endl;

    return true;
}

void application::shutdown(void)
{
    std::cout << "shutdown started" << std::endl;

    delete m_core;
    delete m_fitter_factory_mpfit;
    delete m_fitter_factory_multinest;
    delete m_model_factory_galaxy_2d_cuda;
    delete m_model_factory_galaxy_2d_omp;
    delete m_fitter;
    delete m_model;

    std::cout << "shutdown completed" << std::endl;
}

void application::run(void)
{
    std::map<std::string,float> params = {
        {"vsys",0},
        {"xo",8},
        {"yo",8},
        {"pa",90},
        {"incl",45},
        {"i0",1.0},
        {"r0",2.0},
        {"rt",5.0},
        {"vt",200},
        {"vsig",40}
    };

    std::vector<gbkfit::ndarray*> data = m_model->evaluate(params);
    gbkfit::fits_util::write_to("!foo_flxmap.fits",*data[0]);
    gbkfit::fits_util::write_to("!foo_velmap.fits",*data[1]);
    gbkfit::fits_util::write_to("!foo_sigmap.fits",*data[2]);
}


} // namespace gbkfit_app_cli
