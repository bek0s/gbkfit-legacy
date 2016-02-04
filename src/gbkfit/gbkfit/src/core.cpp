
#include "gbkfit/core.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/fitter.hpp"
#include "gbkfit/model.hpp"
#include "gbkfit/nddataset.hpp"
#include "gbkfit/parameters.hpp"

#include "gbkfit/instrument.hpp"
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

    // Iterate over parameters.
    for(auto& info_ptree_child : info_ptree)
    {
        if(info_ptree_child.first == "param")
        {
            std::string param_name = info_ptree_child.second.get<std::string>("<xmlattr>.name");
            params->add_parameter(param_name);

            // Iterate over parameter options.
            for(auto& item : info_ptree_child.second.get_child("<xmlattr>"))
            {
                std::string option_name = item.first;
                std::string option_value = item.second.data();
                params->get_parameter(param_name).add<std::string>(option_name, option_value);
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

    Dataset* dataset = new Dataset(dataset_type);

    // Iterate over data.
    for(auto& info_ptree_child : info_ptree)
    {
        if(info_ptree_child.first == "data")
        {
            std::string data_type = info_ptree_child.second.get<std::string>("<xmlattr>.type");
            std::string data_file = info_ptree_child.second.get<std::string>("<xmlattr>.file");

            NDArray* data = fits::get_data(data_file);

            dataset->add_data(data_type, data);
        }
    }

    return dataset;
}

std::map<std::string, Dataset*> Core::create_datasets(const std::string& info) const
{
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream, info_ptree);

    Datasets datasets;

    // Iterate over all datasets.
    for(auto& info_ptree_child : info_ptree)
    {
        if(info_ptree_child.first == "dataset")
        {
            std::stringstream dataset_info;
            boost::property_tree::write_xml(dataset_info, info_ptree_child.second);

            Dataset* dataset = create_dataset(dataset_info.str());

            datasets.emplace(dataset->get_type(), dataset);
        }
    }

    return datasets;
}

instrument* Core::create_instrument(const std::string& info) const
{
    // Parse input string as xml.
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    // Read sampling
    float sampling_x = info_ptree.get<float>("sampling.x");
    float sampling_y = info_ptree.get<float>("sampling.y");
    float sampling_z = info_ptree.get<float>("sampling.z");

    // Read psf and lsf type.
    std::string psf_type = info_ptree.get<std::string>("psf.type");
    std::string lsf_type = info_ptree.get<std::string>("lsf.type");

    point_spread_function* psf = nullptr;
    line_spread_function* lsf = nullptr;

    //
    // Read and create psf.
    //

    if      (psf_type == "gaussian")
    {
        float fwhm = info_ptree.get<float>("psf.fwhm");
        psf = new point_spread_function_gaussian(fwhm);
    }
    else if (psf_type == "lorentzian")
    {
        float fwhm = info_ptree.get<float>("psf.fwhm");
        psf = new point_spread_function_lorentzian(fwhm);

    }
    else if (psf_type == "moffat")
    {
        float fwhm = info_ptree.get<float>("psf.fwhm");
        float beta = info_ptree.get<float>("psf.beta",4.765f);
        psf = new point_spread_function_moffat(fwhm,beta);
    }

    //
    // Read and create lsf.
    //

    if (    lsf_type == "gaussian")
    {
        float fwhm = info_ptree.get<float>("lsf.fwhm");
        lsf = new line_spread_function_gaussian(fwhm);
    }
    else if (lsf_type == "lorentzian")
    {
        float fwhm = info_ptree.get<float>("lsf.fwhm");
        lsf = new line_spread_function_lorentzian(fwhm);
    }
    else if (lsf_type == "moffat")
    {
        float fwhm = info_ptree.get<float>("lsf.fwhm");
        float beta = info_ptree.get<float>("lsf.beta",4.765f);
        lsf = new line_spread_function_moffat(fwhm,beta);
    }

    //
    // Create instrument
    //

    instrument* new_instrument = new instrument(sampling_x, sampling_y, sampling_z, psf, lsf);

    return new_instrument;

}

} // namespace gbkfit
