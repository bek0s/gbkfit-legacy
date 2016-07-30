
#include "gbkfit/core.hpp"
#include "gbkfit/dataset.hpp"
#include "gbkfit/dmodel.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/fitter.hpp"
#include "gbkfit/gmodel.hpp"
#include "gbkfit/json.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/params.hpp"
#include "gbkfit/spread_functions.hpp"

namespace gbkfit {

Core::Core(void)
{
}

Core::~Core()
{
    for(auto& dataset : m_datasets)
        delete dataset;

    for(auto& psf : m_psfs)
        delete psf;

    for(auto& lsf : m_lsfs)
        delete lsf;

    for(auto& dmodel : m_dmodels)
        get_dmodel_factory(dmodel->get_type())->destroy(dmodel);

    for(auto& gmodel : m_gmodels)
        get_gmodel_factory(gmodel->get_type())->destroy(gmodel);

    for(auto& fitter : m_fitters)
        get_fitter_factory(fitter->get_type())->destroy(fitter);

    for(auto& parameters : m_parameters)
        delete parameters;
}

void Core::add_dmodel_factory(const DModelFactory* factory)
{
    if (m_dmodel_factories.count(factory->get_type()))
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    m_dmodel_factories[factory->get_type()] = factory;
}

void Core::add_gmodel_factory(const GModelFactory* factory)
{
    if (m_gmodel_factories.count(factory->get_type()))
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    m_gmodel_factories[factory->get_type()] = factory;
}

void Core::add_fitter_factory(const FitterFactory* factory)
{
    m_fitter_factories[factory->get_type()] = factory;
}

Fitter* Core::create_fitter(const std::string& info)
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::string type = info_root.at("type");

    Fitter* fitter = get_fitter_factory(type)->create(info);

    m_fitters.push_back(fitter);

    return fitter;
}

Params* Core::create_parameters(const std::string& info)
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    Params* params = new Params();

    // Iterate over parameters
    for(auto& info_param : info_root)
    {
        Param& param = params->add(info_param.at("name"));

        // Iterate over parameter options
        for (auto it = info_param.begin(); it != info_param.end(); ++it)
        {
            param.add<std::string>(it.key(), it.value().dump());
        }
    }

    m_parameters.push_back(params);

    return params;
}

Dataset* Core::create_dataset(const std::string& info)
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::string type = info_root.at("type").get<std::string>();

    std::string file_dataset_d = info_root.count("data") ? info_root.at("data").get<std::string>() : "";
    std::string file_dataset_e = info_root.count("error") ? info_root.at("error").get<std::string>() : "";
    std::string file_dataset_m = info_root.count("mask") ? info_root.at("mask").get<std::string>() : "";

    std::unique_ptr<NDArray> dataset_d = file_dataset_d.length() ? fits::get_data(file_dataset_d) : nullptr;
    std::unique_ptr<NDArray> dataset_e = file_dataset_e.length() ? fits::get_data(file_dataset_e) : nullptr;
    std::unique_ptr<NDArray> dataset_m = file_dataset_m.length() ? fits::get_data(file_dataset_m) : nullptr;

    Dataset* dataset = new Dataset(type, dataset_d.release(), dataset_e.release(), dataset_m.release());
    m_datasets.push_back(dataset);

    return dataset;
}

std::vector<Dataset*> Core::create_datasets(const std::string& info)
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::vector<Dataset*> datasets;

    for(auto& info_child : info_root)
    {
        std::string foo = info_child.dump();

        Dataset* dataset = create_dataset(foo);
        datasets.push_back(dataset);
    }


    return datasets;
}

LineSpreadFunction* Core::create_line_spread_function(const std::string& info,
                                                      float step)
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::string type = info_root.at("type").get<std::string>();

    LineSpreadFunction* lsf = nullptr;

    if      (type == "gaussian")
    {
        float fwhm = info_root.at("fwhm").get<float>();
        lsf = new LineSpreadFunctionGaussian(fwhm);
    }
    else if (type == "lorentzian")
    {
        float fwhm = info_root.at("fwhm").get<float>();
        lsf = new LineSpreadFunctionLorentzian(fwhm);
    }
    else if (type == "moffat")
    {
        float fwhm = info_root.at("fwhm").get<float>();
        float beta = info_root.at("beta").get<float>();
        lsf = new LineSpreadFunctionMoffat(fwhm, beta);
    }
    else if (type == "array")
    {
        std::string file = info_root.at("file").get<std::string>();
        std::shared_ptr<NDArrayHost> data = fits::get_data(file);
        lsf = new LineSpreadFunctionArray(data->get_host_ptr(),
                                          data->get_shape()[0],
                                          step);
    }
    else
    {
        lsf = new LineSpreadFunctionNone();
    }

    m_lsfs.push_back(lsf);

    return lsf;
}

PointSpreadFunction* Core::create_point_spread_function(const std::string& info,
                                                        float step_x,
                                                        float step_y)
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::string psf_type = info_root.at("type").get<std::string>();

    PointSpreadFunction* psf = nullptr;

    if      (psf_type == "gaussian")
    {
        float fwhm_x = info_root.at("fwhm_x").get<float>();
        float fwhm_y = info_root.at("fwhm_y").get<float>();
        float pa = info_root.at("pa").get<float>();
        psf = new PointSpreadFunctionGaussian(fwhm_x, fwhm_y, pa);
    }
    else if (psf_type == "lorentzian")
    {
        float fwhm_x = info_root.at("fwhm_x").get<float>();
        float fwhm_y = info_root.at("fwhm_y").get<float>();
        float pa = info_root.at("pa").get<float>();
        psf = new PointSpreadFunctionLorentzian(fwhm_x, fwhm_y, pa);

    }
    else if (psf_type == "moffat")
    {
        float fwhm_x = info_root.at("fwhm_x").get<float>();
        float fwhm_y = info_root.at("fwhm_y").get<float>();
        float beta = info_root.at("beta").get<float>();
        float pa = info_root.at("pa").get<float>();
        psf = new PointSpreadFunctionMoffat(fwhm_x, fwhm_y, pa, beta);
    }
    else if (psf_type == "image")
    {
        std::string file = info_root.at("file").get<std::string>();
        std::shared_ptr<NDArrayHost> data = fits::get_data(file);
        psf = new PointSpreadFunctionImage(data->get_host_ptr(),
                                           data->get_shape()[0],
                                           data->get_shape()[1],
                                           step_x,
                                           step_y);
    }
    else
    {
        psf = new PointSpreadFunctionNone();
    }

    m_psfs.push_back(psf);

    return psf;
}

const FitterFactory* Core::get_fitter_factory(const std::string& type) const
{
    auto iter = m_fitter_factories.find(type);
    if (iter == m_fitter_factories.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    return iter->second;
}

DModel* Core::create_dmodel(const std::string& info,
                            const std::vector<int>& size,
                            const std::vector<float>& step,
                            const PointSpreadFunction* psf,
                            const LineSpreadFunction* lsf)
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::string type = info_root.at("type");

    DModel* dmodel = get_dmodel_factory(type)->create(info, size, step, psf, lsf);

    m_dmodels.push_back(dmodel);

    return dmodel;
}


GModel* Core::create_gmodel(const std::string& info)
{
    nlohmann::json info_root = nlohmann::json::parse(info);

    std::string type = info_root.at("type");

    GModel* gmodel = get_gmodel_factory(type)->create(info);

    m_gmodels.push_back(gmodel);

    return gmodel;
}

const DModelFactory* Core::get_dmodel_factory(const std::string& type) const
{
    auto iter = m_dmodel_factories.find(type);

    if (iter == m_dmodel_factories.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    return iter->second;
}

const GModelFactory* Core::get_gmodel_factory(const std::string& type) const
{
    auto iter = m_gmodel_factories.find(type);
    if (iter == m_gmodel_factories.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
    return iter->second;
}



} // namespace gbkfit
