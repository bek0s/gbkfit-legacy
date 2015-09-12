
#include "gbkfit/core.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/fitter.hpp"
#include "gbkfit/model.hpp"
#include "gbkfit/nddataset.hpp"
#include "gbkfit/parameters_fit_info.hpp"

#include "gbkfit/instrument.hpp"
#include "gbkfit/spread_function.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {

void core::add_model_factory(model_factory* factory)
{
    m_model_factories[factory->get_type_name()] = factory;
}

void core::add_fitter_factory(fitter_factory* factory)
{
    m_fitter_factories[factory->get_type_name()] = factory;
}

model* core::create_model(const std::string& info) const
{
    // Parse input string as xml.
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    // Get type.
    std::string type = info_ptree.get<std::string>("type");

    // Get factory for the requested type.
    auto iter = m_model_factories.find(type);

    // Throw an exception if there is no factory available.
    if(iter == m_model_factories.end()) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    // Create factory product with the supplied info and return it.
    return iter->second->create_model(info);
}

fitter* core::create_fitter(const std::string& info) const
{
    // Parse input string as xml.
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    // Get type.
    std::string type = info_ptree.get<std::string>("type");

    // Get factory for the requested type.
    auto iter = m_fitter_factories.find(type);

    // Throw an exception if there is no factory available.
    if(iter == m_fitter_factories.end()){
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    // Create factory product with the supplied info and return it.
    return iter->second->create_fitter(info);
}

parameters_fit_info* core::create_parameters_fit_info(const std::string& info) const
{
    // Parse input string as xml.
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    parameters_fit_info* params_fit_info = new parameters_fit_info();

    // Iterate over...
    for(auto& info_ptree_child : info_ptree)
    {
        // ...parameters.
        if(info_ptree_child.first == "parameter")
        {
            // Get the name of the parameter.
            std::string param_name = info_ptree_child.second.get<std::string>("<xmlattr>.name");

            // Create a new parameter.
            params_fit_info->add_parameter(param_name);

            // Itarate over the rest of the xml attributes
            for(auto& item : info_ptree_child.second.get_child("<xmlattr>"))
            {
                std::string name = item.first;
                if(name != "name")
                {
                    std::string value = info_ptree_child.second.get<std::string>("<xmlattr>." + name);

                    params_fit_info->get_parameter(param_name).add<std::string>(name,value);
                }
            }
        }
    }

    return params_fit_info;
}

nddataset* core::create_dataset(const std::string& info) const
{
    // Parse input string as xml.
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    // Create dataset.
    nddataset* dataset = new nddataset();

    // Iterate over...
    for(auto& info_ptree_child : info_ptree)
    {
        // ...data.
        if(info_ptree_child.first == "data")
        {
            // Get data name and filename.
            std::string data_name = info_ptree_child.second.get<std::string>("<xmlattr>.name");
            std::string data_file = info_ptree_child.second.get_value<std::string>();

            // ...
            std::cout << "Reading data '" << data_name << "'..." << std::endl;

            // Read the file.
            ndarray* data = fits::get_data(data_file);

            // Add the file to the dataset.
            dataset->add_data(data_name,data);
        }
    }

    return dataset;
}

std::map<std::string,nddataset*> core::create_datasets(const std::string& info) const
{   
    // Parse input string as xml.
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    // Create dataset map.
    std::map<std::string,nddataset*> datasets;

    // Iterate over...
    for(auto& info_ptree_child : info_ptree)
    {
        // ...datasets.
        if(info_ptree_child.first == "dataset")
        {
            // Get dataset name.
            std::string dataset_name = info_ptree_child.second.get<std::string>("<xmlattr>.name");

            // ...
            std::cout << "Reading dataset '" << dataset_name << "'..." << std::endl;

            // Prepare xml info for dataset.
            std::stringstream dataset_info;
            boost::property_tree::write_xml(dataset_info,info_ptree_child.second);

            // Create the dataset.
            nddataset* dataset = create_dataset(dataset_info.str());

            // Add the dataset to the map.
            datasets.emplace(dataset_name,dataset);
        }
    }

    return datasets;
}

instrument* core::create_instrument(const std::string& info) const
{
    // Parse input string as xml.
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    // Read sampling
    float sampling_x = info_ptree.get<float>("sampling_x");
    float sampling_y = info_ptree.get<float>("sampling_y");
    float sampling_z = info_ptree.get<float>("sampling_z");

    // Read psf and lsf type.
    std::string psf_type = info_ptree.get<std::string>("psf.<xmlattr>.type");
    std::string lsf_type = info_ptree.get<std::string>("lsf.<xmlattr>.type");

    point_spread_function* psf = nullptr;
    line_spread_function* lsf = nullptr;

    //
    // Read and create psf.
    //

    if      (psf_type == "gaussian")
    {
        float fwhm = info_ptree.get<float>("psf.<xmlattr>.fwhm");
        psf = new point_spread_function_gaussian(fwhm);
    }
    else if (psf_type == "lorentzian")
    {
        float fwhm = info_ptree.get<float>("psf.<xmlattr>.fwhm");
        psf = new point_spread_function_lorentzian(fwhm);

    }
    else if (psf_type == "moffat")
    {
        float fwhm = info_ptree.get<float>("psf.<xmlattr>.fwhm");
        float beta = info_ptree.get<float>("psf.<xmlattr>.beta",4.765f);
        psf = new point_spread_function_moffat(fwhm,beta);
    }

    //
    // Read and create lsf.
    //

    if (    lsf_type == "gaussian")
    {
        float fwhm = info_ptree.get<float>("lsf.<xmlattr>.fwhm");
        lsf = new line_spread_function_gaussian(fwhm);
    }
    else if (lsf_type == "lorentzian")
    {
        float fwhm = info_ptree.get<float>("lsf.<xmlattr>.fwhm");
        lsf = new line_spread_function_lorentzian(fwhm);
    }
    else if (lsf_type == "moffat")
    {
        float fwhm = info_ptree.get<float>("lsf.<xmlattr>.fwhm");
        float beta = info_ptree.get<float>("lsf.<xmlattr>.beta",4.765f);
        lsf = new line_spread_function_moffat(fwhm,beta);
    }

    //
    // Create instrument
    //

    instrument* new_instrument = new instrument(sampling_x, sampling_y, sampling_z, psf, lsf);

    return new_instrument;

}

} // namespace gbkfit
