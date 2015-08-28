
#include "gbkfit/core.hpp"
#include "gbkfit/fits.hpp"
#include "gbkfit/fitter.hpp"
#include "gbkfit/model.hpp"
#include "gbkfit/nddataset.hpp"

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
            std::string data_file = info_ptree_child.second.get<std::string>("<xmlattr>.value");

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
            std::string dataset_name = info_ptree_child.second.get<std::string>("name");

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


} // namespace gbkfit
