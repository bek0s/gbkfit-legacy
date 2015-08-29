
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

std::string get_error_string(int code)
{
    std::string str_code;
    std::string str_desc;

    switch (code)
    {
    // Success status codes.
    case MP_OK_CHI:
        str_code = "MP_OK_CHI";
        str_desc = "Convergence in chi-square value.";
        break;
    case MP_OK_PAR:
        str_code = "MP_OK_CHI";
        str_desc = "Convergence in parameter value.";
        break;
    case MP_OK_BOTH:
        str_code = "MP_OK_BOTH";
        str_desc = "Convergence in both chi-square and parameter value.";
        break;
    case MP_OK_DIR:
        str_code = "MP_OK_DIR";
        str_desc = "Convergence in orthogonality.";
        break;
    case MP_MAXITER:
        str_code = "MP_MAXITER";
        str_desc = "Maximum number of iterations reached.";
        break;
    case MP_FTOL:
        str_code = "MP_FTOL";
        str_desc = "ftol is too small; no further improvement.";
        break;
    case MP_XTOL:
        str_code = "MP_XTOL";
        str_desc = "xtol is too small; no further improvement.";
        break;
    case MP_GTOL:
        str_code = "MP_GTOL";
        str_desc = "gtol is too small; no further improvement.";
        break;
    // Error status codes.
    case MP_ERR_INPUT:
        // This is not used by mpfit, I don't know why it exits.
        str_code = "MP_ERR_INPUT";
        str_desc = "General input parameter error.";
        break;
    case MP_ERR_NAN:
        str_code = "MP_ERR_NAN";
        str_desc = "User function produced non-finite values.";
        break;
    case MP_ERR_FUNC:
        str_code = "MP_ERR_FUNC";
        str_desc = "No user function was supplied.";
        break;
    case MP_ERR_NPOINTS:
        str_code = "MP_ERR_NPOINTS";
        str_desc = "No user data points were supplied.";
        break;
    case MP_ERR_NFREE:
        str_code = "MP_ERR_NFREE";
        str_desc = "No free parameters.";
        break;
    case MP_ERR_MEMORY:
        str_code = "MP_ERR_MEMORY";
        str_desc = "Memory allocation error.";
        break;
    case MP_ERR_INITBOUNDS:
        str_code = "MP_ERR_INITBOUNDS";
        str_desc = "Initial values inconsistent with constraints.";
        break;
    case MP_ERR_BOUNDS:
        str_code = "MP_ERR_BOUNDS";
        str_desc = "Initial constraints inconsistent.";
        break;
    case MP_ERR_PARAM:
        str_code = "MP_ERR_PARAM";
        str_desc = "General input parameter error.";
        break;
    case MP_ERR_DOF:
        str_code = "MP_ERR_DOF";
        str_desc = "Not enough degrees of freedom.";
        break;

    default:
        str_code = "UNKNOWN_ERROR";
        str_desc = "Unknown error.";
        break;
    }

    return str_code + ", " + str_desc;
}

#define MP_ERR_INPUT (0)         /* General input parameter error */
#define MP_ERR_NAN (-16)         /* User function produced non-finite values */
#define MP_ERR_FUNC (-17)        /* No user function was supplied */
#define MP_ERR_NPOINTS (-18)     /* No user data points were supplied */
#define MP_ERR_NFREE (-19)       /* No free parameters */
#define MP_ERR_MEMORY (-20)      /* Memory allocation error */
#define MP_ERR_INITBOUNDS (-21)  /* Initial values inconsistent w constraints*/
#define MP_ERR_BOUNDS (-22)      /* Initial constraints inconsistent */
#define MP_ERR_PARAM (-23)       /* General input parameter error */
#define MP_ERR_DOF (-24)         /* Not enough degrees of freedom */

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
    int measurement_count = 0;

    // Iterate over the supplied datasets.
    for(auto& dataset : datasets)
    {
        std::string dataset_name = std::get<0>(dataset);

        // Get input data pointer shortcuts for convenience.
        ndarray* old_data_dat = std::get<1>(dataset)->get_data("data");
        ndarray* old_data_msk = std::get<1>(dataset)->get_data("mask");
        ndarray* old_data_err = std::get<1>(dataset)->get_data("error");

        // Allocate model data memory on the host.
        data_map_mdl[dataset_name] = new ndarray_host_new(old_data_dat->get_shape());

        // Copy data to the host.
        data_map_dat[dataset_name] = new ndarray_host_new(*old_data_dat);
        data_map_msk[dataset_name] = new ndarray_host_new(*old_data_msk);
        data_map_err[dataset_name] = new ndarray_host_new(*old_data_err);

        measurement_count += old_data_dat->get_shape().get_dim_length_product();
    }

    //
    // Get model parameters. Always use this order for the parameters!
    //

    std::vector<std::string> model_param_names = model->get_parameter_names();

    //
    // Allocate an array for model parameters and give initial values.
    //

    std::vector<double> model_param_values;
    for(auto& param_name : model_param_names)
        model_param_values.push_back(params_info.get_parameter(param_name).get<float>("init"));

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
    mpfit_config_info.maxiter = 2000; // TODO

    //
    // Setup mpfit's results struct.
    //

    mp_result mpfit_result_info;
    std::memset(&mpfit_result_info,0,sizeof(mpfit_result_info));
    mpfit_result_info.xerror = new double[model_param_names.size()];
    mpfit_result_info.covar = new double[model_param_names.size()*model_param_names.size()];

    //
    // Create and populate user data.
    //

    mpfit_user_data mpfit_udata;
    mpfit_udata.model = model;
    mpfit_udata.data_map_mdl = data_map_mdl;
    mpfit_udata.data_map_dat = data_map_dat;
    mpfit_udata.data_map_msk = data_map_msk;
    mpfit_udata.data_map_err = data_map_err;

    //
    // Time to fit! Woohoo!
    //

    int mpfit_result = ::mpfit(mpfit_callback,
                               measurement_count,
                               model_param_values.size(),
                               model_param_values.data(),
                               mpfit_params_info.data(),
                               &mpfit_config_info,
                               &mpfit_udata,
                               &mpfit_result_info);

    std::cout << "Optimization completed [code: " << get_error_string(mpfit_result) << "]." << std::endl;
    std::cout << "Initial chi-square: " << mpfit_result_info.orignorm << std::endl;
    std::cout << "Final chi-square: " << mpfit_result_info.bestnorm << std::endl;

    //
    // Extract fitted model parameters.
    //

    for (std::size_t i = 0; i < model_param_names.size(); ++i)
    {
        std::string param_name = model_param_names[i];
        float param_best = model_param_values[i];
        float param_error = mpfit_result_info.xerror[i];

        std::cout << "Fitted param:"
                  << " name=" << std::setw(4) << param_name
                  << " best=" << std::setw(8) << param_best
                  << " error=" << std::setw(8) << param_error << std::endl;
    }

    //
    // Perform clean up
    //

    delete [] mpfit_result_info.xerror;
    delete [] mpfit_result_info.covar;

    for(auto& mpfit_param_info : mpfit_params_info)
        delete [] mpfit_param_info.parname;

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
