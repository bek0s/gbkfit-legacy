
#include "gbkfit/fitters/mpfit/fitter_mpfit.hpp"
#include "gbkfit/fitters/mpfit/mpfit/mpfit.h"
#include "gbkfit/model.hpp"
#include "gbkfit/model_parameters_fit_info.hpp"
#include "gbkfit/nddataset.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/utility.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {
namespace fitters {
namespace mpfit {

struct mpfit_user_data
{
    gbkfit::model* model;
    std::map<std::string,ndarray_host*> data_map_mdl;
    std::map<std::string,ndarray_host*> data_map_dat;
    std::map<std::string,ndarray_host*> data_map_msk;
    std::map<std::string,ndarray_host*> data_map_err;
}; // struct mpfit_user_data

int mpfit_callback(int num_measurements, int num_parameters, double* parameters, double* measurements, double** derivatives, void* user_data)
{
    mpfit_user_data* udata = reinterpret_cast<mpfit_user_data*>(user_data);


    std::vector<float> params(parameters,parameters+num_parameters);
    std::vector<std::string> param_name;

    auto params_map = gbkfit::vectors_to_map(param_name, params);


    std::vector<ndarray*> model_data = udata->model->evaluate(params_map);


    /*
    (void)derivatives; // unused

    std::size_t ipar;

    // retrieve user data
    mpfit_user_data* udata = reinterpret_cast<mpfit_user_data*>(user_data);

#if 0
    // print the current set of parameters
    ipar = 0;
    std::cout << std::endl;
    for(model_parameter_set::const_iterator iter = udata->params.begin(); iter != udata->params.end(); ++iter) {
        std::cout << (*iter).first << ": " << parameters[ipar++] << std::endl;
    }
#endif

    // validate parameters
    if(std::any_of(parameters,parameters+num_parameters,[](float num){return !std::isfinite(num);})) {
        throw std::runtime_error("mpfit_callback: non-finite parameter value found");
    }

    // build parameter map using the current set of the parameters
    ipar = 0;
    std::map<std::string,float> parameter_map;
    for(model_parameter_set::const_iterator iter = udata->params.begin(); iter != udata->params.end(); ++iter) {
        parameter_map[iter->first] = static_cast<float>(parameters[ipar++]);
    }

    // evaluate model and retrieve model data
    udata->model->evaluate(parameter_map,udata->data_mdl);

    // calculate weighted residuals
    for(int i = 0; i < num_measurements; ++i) {
        udata->data_res[i] = udata->data_obs_msk[i] > 0 ? (udata->data_obs[i]-udata->data_mdl[i])/udata->data_obs_err[i] : 0;
    }

    // validate residuals
    if(std::any_of(std::begin(udata->data_res),std::end(udata->data_res),[](float num){return !std::isfinite(num);})) {
        throw std::runtime_error("mpfit_callback: non-finite residual value found");
    }

    // feed mpfit with residuals
    std::copy(std::begin(udata->data_res),std::end(udata->data_res),measurements);

    */
    // return success
    return 0;
}

fitter_mpfit::fitter_mpfit(void)
{
}

fitter_mpfit::~fitter_mpfit()
{
}

const std::string& fitter_mpfit::get_type_name(void) const
{
    return fitter_factory_mpfit::FACTORY_TYPE_NAME;
}

void fitter_mpfit::fit(model* model, const std::map<std::string,nddataset*>& datasets, model_parameters_fit_info& params_info) const
{
    //
    // Make a copy of the required data on the host.
    //

    std::map<std::string,ndarray_host*> data_map_mdl;
    std::map<std::string,ndarray_host*> data_map_dat;
    std::map<std::string,ndarray_host*> data_map_msk;
    std::map<std::string,ndarray_host*> data_map_err;

    // Iterate over the supplied datasets.
    for(auto& dataset : datasets)
    {
        std::string dataset_name = std::get<0>(dataset);

        // Get input data pointer shortcuts for convenience.
        ndarray* old_data_dat = std::get<1>(dataset)->get_data("data");
        ndarray* old_data_msk = std::get<1>(dataset)->get_data("mask");
        ndarray* old_data_err = std::get<1>(dataset)->get_data("error");

        // Allocate model memory data on the host.
        data_map_mdl[dataset_name] = new ndarray_host_new(old_data_dat->get_shape());

        // Copy data to the host.
        data_map_dat[dataset_name] = new ndarray_host_new(*old_data_dat);
        data_map_msk[dataset_name] = new ndarray_host_new(*old_data_msk);
        data_map_err[dataset_name] = new ndarray_host_new(*old_data_err);
    }

    //
    // Get model parameters. Use this order when... you need an order!
    //

    std::vector<std::string> model_param_names = model->get_parameter_names();

    //
    // Setup mpfit's per-parameter configuration.
    //

    std::vector<mp_par> mpfit_params_info;
    for(auto& param_name : model_param_names)
    {
        mp_par mpfit_param_info;
        std::memset(&mpfit_param_info,0,sizeof(mpfit_param_info));
        mpfit_param_info.step = params_info.get_parameter(param_name).get<float>("step",0);
        mpfit_param_info.side = params_info.get_parameter(param_name).get<int>("side",0);
        mpfit_param_info.fixed = params_info.get_parameter(param_name).get<bool>("fixed",0);
        mpfit_param_info.limited[0] = params_info.get_parameter(param_name).has("min");
        mpfit_param_info.limited[1] = params_info.get_parameter(param_name).has("max");
        mpfit_param_info.limits[0] = params_info.get_parameter(param_name).get<float>("min",-std::numeric_limits<float>::max());
        mpfit_param_info.limits[1] = params_info.get_parameter(param_name).get<float>("max",+std::numeric_limits<float>::max());
        mpfit_param_info.parname = new char[param_name.size()+1];
        std::strcpy(mpfit_param_info.parname,param_name.c_str());
        mpfit_params_info.push_back(mpfit_param_info);
    }

    //
    // Setup mpfit's global configuration.
    //

    mp_config mpfit_config_info;
    std::memset(&mpfit_config_info,0,sizeof(mpfit_config_info));
    mpfit_config_info.maxiter = 1000;

    //
    // Setup mpfit's results struct.
    //

    mp_result mpfit_result_info;
    std::memset(&mpfit_result_info,0,sizeof(mpfit_result_info));
    mpfit_result_info.xerror = new double[model_param_names.size()];

    //
    // Prepare model parameter initial values.
    //

    std::vector<double> model_param_init;
    for(auto& param_name : model_param_names)
    {
        model_param_init.push_back(params_info.get_parameter(param_name).get<float>("init"));
    }


    //
    // Create and populate user data.
    //

    mpfit_user_data udata;
    udata.model = model;
    udata.data_map_mdl = data_map_mdl;
    udata.data_map_dat = data_map_dat;
    udata.data_map_msk = data_map_msk;
    udata.data_map_err = data_map_err;


    /*


    // fit!
    int mpfit_result = mpfit(mpfit_callback,
                             udata.model->get_model_data_length(),
                             model_params.size(),
                             model_param_values.data(),
                             mpfit_params_info.data(),
                             &mpfit_config_info,
                             &udata,
                             &mpfit_result_info);
    (void)mpfit_result;

    // copy the results from the fitter to the model parameter info object
    std::size_t ipar = 0;
    for(auto& param : model_params)
    {
        param.second.best = model_param_values[ipar];
        param.second.stddev = mpfit_result_info.xerror[ipar];
        ipar++;
    }

    std::cout << mpfit_result_info.bestnorm << std::endl;

    // cleanup
    delete [] mpfit_result_info.xerror;
    */


    //
    // Perform clean up
    //

    for(auto& data : data_map_mdl) delete std::get<1>(data);
    for(auto& data : data_map_dat) delete std::get<1>(data);
    for(auto& data : data_map_msk) delete std::get<1>(data);
    for(auto& data : data_map_err) delete std::get<1>(data);
}

const std::string fitter_factory_mpfit::FACTORY_TYPE_NAME = "gbkfit.fitters.mpfit";

fitter_factory_mpfit::fitter_factory_mpfit(void)
{
}

fitter_factory_mpfit::~fitter_factory_mpfit()
{
}

const std::string& fitter_factory_mpfit::get_type_name(void) const
{
    return FACTORY_TYPE_NAME;
}

fitter* fitter_factory_mpfit::create_fitter(const std::string& info) const
{
    // Parse input string as xml.
    std::stringstream info_stream(info);
    boost::property_tree::ptree info_ptree;
    boost::property_tree::read_xml(info_stream,info_ptree);

    return new fitter_mpfit();
}

} // namepsece mpfit
} // namespace fitters
} // namespace gbkfit
