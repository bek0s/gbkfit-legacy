
#include "gbkfit/core.hpp"
#include "gbkfit/model.hpp"
#include "gbkfit/fitter.hpp"
#include "gbkfit/fits_util.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {


void core::add_model_factory(model_factory* factory)
{
    m_model_factories[factory->get_type_name()] = factory;
}

model* core::create_model(std::stringstream& info) const
{
    boost::property_tree::ptree ptree_info;
    boost::property_tree::read_xml(info,ptree_info);

    std::string type = ptree_info.get<std::string>("type");

    // get factory for the requested type
    auto iter = m_model_factories.find(type);
    if(iter == m_model_factories.end()){
        throw std::runtime_error("Factory for the requested model type not found.");
    }

    // create
    return iter->second->create_model(info);
}

void core::add_fitter_factory(fitter_factory* factory)
{
    m_fitter_factories[factory->get_type_name()] = factory;
}

fitter* core::create_fitter(std::stringstream& info) const
{
    boost::property_tree::ptree ptree_info;
    boost::property_tree::read_xml(info,ptree_info);

    std::string type = ptree_info.get<std::string>("type");

    // get factory for the requested type
    auto iter = m_fitter_factories.find(type);
    if(iter == m_fitter_factories.end()){
        throw std::runtime_error("Factory for the requested fitter type not found.");
    }

    // create
    return iter->second->create_fitter(info);
}

nddataset* core::create_dataset(std::stringstream& info) const
{
    nddataset* dataset = new nddataset();
    boost::property_tree::ptree ptree_info;
    boost::property_tree::read_xml(info,ptree_info);

    for(auto& ptree_data : ptree_info)
    {
        std::string data_name = ptree_data.second.get<std::string>("<xmlattr>.name");
        std::string data_file = ptree_data.second.get<std::string>("<xmlattr>.value");
        ndarray* data = fits_util::get_data(data_file);

        dataset->get().emplace(data_name,data);

    }

    return nullptr;
}

std::map<std::string,nddataset*> core::create_datasets(std::stringstream& info) const
{
    std::map<std::string,nddataset*> datasets;
    boost::property_tree::ptree ptree_info;
    boost::property_tree::read_xml(info,ptree_info);

    for(auto& ptree_dataset : ptree_info)
    {
        std::stringstream dataset_info;
        boost::property_tree::write_xml(dataset_info,ptree_dataset.second);
        std::string name = ptree_dataset.second.get<std::string>("<xmlattr>.name");
        nddataset* dataset = create_dataset(dataset_info);
        datasets.emplace(name,dataset);
    }

    return datasets;
}


} // namespace gbkfit
