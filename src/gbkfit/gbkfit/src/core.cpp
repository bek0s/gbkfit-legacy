
#include "gbkfit/core.hpp"
#include "gbkfit/dataset.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/fitter.hpp"
#include "gbkfit/instrument.hpp"
#include "gbkfit/model.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/parameters.hpp"
#include "gbkfit/spread_functions.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {

void Core::add_model_factory(ModelFactory* factory)
{
    m_model_factories[factory->get_type_name()] = factory;
}

void Core::add_fitter_factory(FitterFactory* factory)
{
    m_fitter_factories[factory->get_type_name()] = factory;
}

Model* Core::create_model(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream, info_ptree);

    std::string model_type = info_ptree.get<std::string>("type");

    auto iter = m_model_factories.find(model_type);
    if (iter == m_model_factories.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return iter->second->create_model(info);
}

Fitter* Core::create_fitter(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream, info_ptree);

    std::string fitter_type = info_ptree.get<std::string>("type");

    auto iter = m_fitter_factories.find(fitter_type);
    if (iter == m_fitter_factories.end()){
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return iter->second->create_fitter(info);
}

Parameters* Core::create_parameters(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream, info_ptree);

    Parameters* params = new Parameters();

    // Iterate over parameters
    for(auto& info_ptree_child : info_ptree)
    {
        if(info_ptree_child.first == "param")
        {
            std::string param_name = info_ptree_child.second.get<std::string>("<xmlattr>.name");
            params->add_parameter(param_name);

            // Iterate over parameter options
            for(auto& option : info_ptree_child.second.get_child("<xmlattr>"))
            {
                params->get_parameter(param_name).add<std::string>(option.first, option.second.data());
            }
        }
    }

    return params;
}

Dataset* Core::create_dataset(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream, info_ptree);

    std::string dataset_type = info_ptree.get<std::string>("type");
    std::string file_dataset_d = info_ptree.get<std::string>("data", "");
    std::string file_dataset_e = info_ptree.get<std::string>("errors", "");
    std::string file_dataset_m = info_ptree.get<std::string>("mask", "");
    std::unique_ptr<NDArray> dataset_d = file_dataset_d.length() ? fits::get_data(file_dataset_d) : nullptr;
    std::unique_ptr<NDArray> dataset_e = file_dataset_e.length() ? fits::get_data(file_dataset_e) : nullptr;
    std::unique_ptr<NDArray> dataset_m = file_dataset_m.length() ? fits::get_data(file_dataset_m) : nullptr;

    return new Dataset(dataset_type, dataset_d.release(), dataset_e.release(), dataset_m.release());
}

std::map<std::string, Dataset*> Core::create_datasets(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream, info_ptree);

    std::map<std::string, Dataset*> datasets;

    // Iterate over datasets
    for(auto& info_ptree_child : info_ptree)
    {
        if(info_ptree_child.first == "dataset")
        {
            std::stringstream dataset_info;
            boost::property_tree::write_xml(dataset_info, info_ptree_child.second);

            Dataset* dataset = create_dataset(dataset_info.str());

            datasets.emplace(dataset->get_name(), dataset);
        }
    }

    return datasets;
}

Instrument* Core::create_instrument(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    float sampling_x = info_ptree.get<float>("sampling.x");
    float sampling_y = info_ptree.get<float>("sampling.y");
    float sampling_z = info_ptree.get<float>("sampling.z");

    std::string psf_type = info_ptree.get<std::string>("psf.type");
    std::string lsf_type = info_ptree.get<std::string>("lsf.type");

    PointSpreadFunction* psf = nullptr;
    LineSpreadFunction* lsf = nullptr;

    //
    // Create psf
    //

    if      (psf_type == "gaussian")
    {
        float fwhm_x = info_ptree.get<float>("psf.fwhm_x");
        float fwhm_y = info_ptree.get<float>("psf.fwhm_y");
        float pa = info_ptree.get<float>("psf.pa");

        psf = new PointSpreadFunctionGaussian(fwhm_x, fwhm_y, pa);
    }
    else if (psf_type == "lorentzian")
    {
        float fwhm_x = info_ptree.get<float>("psf.fwhm_x");
        float fwhm_y = info_ptree.get<float>("psf.fwhm_y");
        float pa = info_ptree.get<float>("psf.pa");

        psf = new PointSpreadFunctionLorentzian(fwhm_x, fwhm_y, pa);

    }
    else if (psf_type == "moffat")
    {
        float fwhm_x = info_ptree.get<float>("psf.fwhm_x");
        float fwhm_y = info_ptree.get<float>("psf.fwhm_y");
        float beta = info_ptree.get<float>("psf.beta");
        float pa = info_ptree.get<float>("psf.pa");

        psf = new PointSpreadFunctionMoffat(fwhm_x, fwhm_y, pa, beta);
    }
    else if (psf_type == "image")
    {
        std::string file = info_ptree.get<std::string>("psf.file");

        std::shared_ptr<NDArrayHost> data = fits::get_data(file);

        psf = new PointSpreadFunctionImage(data->get_host_ptr(),
                                           data->get_shape()[0],
                                           data->get_shape()[1],
                                           sampling_x,
                                           sampling_y);
    }

    //
    // Create lsf
    //

    if      (lsf_type == "gaussian")
    {
        float fwhm = info_ptree.get<float>("lsf.fwhm");

        lsf = new LineSpreadFunctionGaussian(fwhm);
    }
    else if (lsf_type == "lorentzian")
    {
        float fwhm = info_ptree.get<float>("lsf.fwhm");

        lsf = new LineSpreadFunctionLorentzian(fwhm);
    }
    else if (lsf_type == "moffat")
    {
        float fwhm = info_ptree.get<float>("lsf.fwhm");
        float beta = info_ptree.get<float>("lsf.beta");

        lsf = new LineSpreadFunctionMoffat(fwhm, beta);
    }
    else if (lsf_type == "array")
    {
        std::string file = info_ptree.get<std::string>("lsf.file");

        std::shared_ptr<NDArrayHost> data = fits::get_data(file);

        lsf = new LineSpreadFunctionArray(data->get_host_ptr(),
                                          data->get_shape()[0],
                                          sampling_z);
    }

    //
    // Create instrument
    //

    Instrument* new_instrument = new Instrument(sampling_x, sampling_y, sampling_z, psf, lsf);

    return new_instrument;

}

} // namespace gbkfit
