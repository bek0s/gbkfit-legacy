
#include "gbkfit/models/galaxy_2d/model_factory_galaxy_2d.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {
namespace models {
namespace galaxy_2d {


model_factory_galaxy_2d::model_factory_galaxy_2d(void)
    : model_factory()
{
}

model_factory_galaxy_2d::~model_factory_galaxy_2d()
{
}

void model_factory_galaxy_2d::read_parameters(std::stringstream& info) const
{
    boost::property_tree::ptree ptree;
    boost::property_tree::read_xml(info,ptree);


}


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
