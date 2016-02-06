
#include "gbkfit/fitters/mpfit/fitter_mpfit.hpp"
#include <mpfit.h>

#include "gbkfit/fitters/mpfit/fitter_factory_mpfit.hpp"

#include "gbkfit/model.hpp"
#include "gbkfit/nddataset.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/parameters.hpp"
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

struct mpfit_user_data
{
    gbkfit::Model* model;
    std::vector<std::string> param_names;
    std::vector<std::string> dataset_names;
    std::map<std::string,NDArrayHost*> data_map_dat;
    std::map<std::string,NDArrayHost*> data_map_msk;
    std::map<std::string,NDArrayHost*> data_map_err;
    std::map<std::string,NDArrayHost*> data_map_mdl;
}; // struct mpfit_user_data

int mpfit_callback(int num_measurements, int num_parameters, double* parameters, double* measurements, double** derivatives, void* user_data)
{
    (void)num_measurements;
    (void)derivatives;

#if 0
    //
    // Make sure we have finite parameters.
    //
    if(std::any_of(parameters,parameters+num_parameters,[](float num){return !std::isfinite(num);})) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
#endif

    //
    // Get a convenience pointer to the user data.
    //

    mpfit_user_data* udata = reinterpret_cast<mpfit_user_data*>(user_data);

    //
    // Build parameter map with the current set of parameter values.
    //

    const std::vector<float> param_values(parameters,parameters+num_parameters);
    const std::map<std::string,float> param_map = gbkfit::vectors_to_map(udata->param_names, param_values);

#if 1
    // Provide debug information about the current set of parameters.
    std::cout << gbkfit::to_string(param_map) << std::endl;
#endif

    //
    // Evaluate model with the current set of parameter values.
    //

    std::map<std::string,NDArray*> model_data = udata->model->evaluate(param_map);

    //
    // Calculate residuals between the model and the (available) data.
    //

    int measurements_offset = 0;
    for(auto& dataset_name : udata->dataset_names)
    {
        // Get number of measurements in the current data.
        int data_size = udata->data_map_dat[dataset_name]->get_shape().get_dim_length_product();

        // Create convenience pointers.
        float* data_dat = udata->data_map_dat[dataset_name]->get_host_ptr();
        float* data_msk = udata->data_map_msk[dataset_name]->get_host_ptr();
        float* data_err = udata->data_map_err[dataset_name]->get_host_ptr();
        float* data_mdl = udata->data_map_mdl[dataset_name]->get_host_ptr();

        // Copy model data on the host.
        model_data[dataset_name]->read_data(data_mdl);

        // Calculate residuals and place them to mpfit measurements array.
        for(int i = 0; i < data_size; ++i) {
            measurements[measurements_offset+i] = data_msk[i] > 0 ? (data_dat[i]-data_mdl[i])/data_err[i] : 0;
        }

        // Keep track of the offset needed for the measurements array.
        measurements_offset += data_size;
    }

#if 0
    //
    // Make sure we have finite residuals.
    //
    if(std::any_of(measurements,measurements+num_measurements,[](float num){return !finite(num);})) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
#endif

    // Return success.
    return 0;
}

const double FitterMpfit::DEFAULT_FTOL = 0;
const double FitterMpfit::DEFAULT_XTOL = 0;
const double FitterMpfit::DEFAULT_GTOL = 0;
const double FitterMpfit::DEFAULT_EPSFCN = 0;
const double FitterMpfit::DEFAULT_STEPFACTOR = 0;
const double FitterMpfit::DEFAULT_COVTOL = 0;
const int FitterMpfit::DEFAULT_MAXITER = 2000;
const int FitterMpfit::DEFAULT_MAXFEV = 0;
const int FitterMpfit::DEFAULT_NPRINT = 1;
const int FitterMpfit::DEFAULT_DOUSERSCALE = 0;
const int FitterMpfit::DEFAULT_NOFINITECHECK = 0;

FitterMpfit::FitterMpfit(void)
    : m_ftol(DEFAULT_FTOL)
    , m_xtol(DEFAULT_XTOL)
    , m_gtol(DEFAULT_GTOL)
    , m_epsfcn(DEFAULT_EPSFCN)
    , m_stepfactor(DEFAULT_STEPFACTOR)
    , m_covtol(DEFAULT_COVTOL)
    , m_maxiter(DEFAULT_MAXITER)
    , m_maxfev(DEFAULT_MAXFEV)
    , m_nprint(DEFAULT_NPRINT)
    , m_douserscale(DEFAULT_DOUSERSCALE)
    , m_nofinitecheck(DEFAULT_NOFINITECHECK)
{
}

FitterMpfit::~FitterMpfit()
{
}

const std::string& FitterMpfit::get_type_name(void) const
{
    return FitterFactoryMpfit::FACTORY_TYPE_NAME;
}

void FitterMpfit::fit(Model* model, const std::map<std::string,Dataset*>& datasets, Parameters& params_info) const
{
    //
    // Perform the necessary memory allocations and copy the datasets on the host.
    //

    int measurement_count = 0;
    std::vector<std::string> dataset_names;
    std::map<std::string,NDArrayHost*> data_map_dat;
    std::map<std::string,NDArrayHost*> data_map_msk;
    std::map<std::string,NDArrayHost*> data_map_err;
    std::map<std::string,NDArrayHost*> data_map_mdl;

    // Iterate over the input datasets.
    for(auto& dataset : datasets)
    {
        // Get the dataset name and store it in a vector, we will use it later for convenience.
        std::string dataset_name = std::get<0>(dataset);
        dataset_names.push_back(dataset_name);

        // Get input data pointer shortcuts for convenience.
        NDArray* input_data_dat = std::get<1>(dataset)->get_data("data");
        NDArray* input_data_msk = std::get<1>(dataset)->get_data("mask");
        NDArray* input_data_err = std::get<1>(dataset)->get_data("error");

        // Allocate host memory for the input data and the model.
        data_map_dat[dataset_name] = new NDArrayHost(input_data_dat->get_shape());
        data_map_msk[dataset_name] = new NDArrayHost(input_data_msk->get_shape());
        data_map_err[dataset_name] = new NDArrayHost(input_data_err->get_shape());
        data_map_mdl[dataset_name] = new NDArrayHost(input_data_dat->get_shape());

        // Copy data to the host.
        data_map_dat[dataset_name]->write_data(input_data_dat);
        data_map_msk[dataset_name]->write_data(input_data_msk);
        data_map_err[dataset_name]->write_data(input_data_err);

        // Keep track the number of measurements across all input data
        measurement_count += input_data_dat->get_shape().get_dim_length_product();
    }

    //
    // Get model parameters. Always use this order for the parameters!
    //

    std::vector<std::string> param_names = model->get_parameter_names();

    //
    // Allocate an array for model parameters and give initial values.
    //

    std::vector<double> param_values;
    for(auto& param_name : param_names)
        param_values.push_back(params_info.get_parameter(param_name).get<float>("value"));

    //
    // Setup mpfit's per-parameter configuration.
    //

    std::vector<mp_par> mpfit_params_info;
    for(auto& param_name : param_names)
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
    mpfit_config_info.ftol = m_ftol;
    mpfit_config_info.xtol = m_xtol;
    mpfit_config_info.gtol = m_gtol;
    mpfit_config_info.epsfcn = m_epsfcn;
    mpfit_config_info.stepfactor = m_stepfactor;
    mpfit_config_info.covtol = m_covtol;
    mpfit_config_info.maxiter = m_maxiter;
    mpfit_config_info.maxfev = m_maxfev;
    mpfit_config_info.nprint = m_nprint;
    mpfit_config_info.douserscale = m_douserscale;
    mpfit_config_info.nofinitecheck = m_nofinitecheck;

    //
    // Setup mpfit's results struct.
    //

    mp_result mpfit_result_info;
    std::memset(&mpfit_result_info,0,sizeof(mpfit_result_info));
    mpfit_result_info.xerror = new double[param_names.size()];
    mpfit_result_info.covar = new double[param_names.size()*param_names.size()];

    //
    // Create and populate user data.
    //

    mpfit_user_data mpfit_udata;
    mpfit_udata.model = model;
    mpfit_udata.param_names = param_names;
    mpfit_udata.dataset_names = dataset_names;
    mpfit_udata.data_map_dat = data_map_dat;
    mpfit_udata.data_map_msk = data_map_msk;
    mpfit_udata.data_map_err = data_map_err;
    mpfit_udata.data_map_mdl = data_map_mdl;

    //
    // Time to fit! Woohoo!
    //

    int mpfit_result = ::mpfit(mpfit_callback,
                               measurement_count,
                               param_values.size(),
                               param_values.data(),
                               mpfit_params_info.data(),
                               &mpfit_config_info,
                               &mpfit_udata,
                               &mpfit_result_info);

    std::cout << "Optimization completed [mpfit code: " << get_error_string(mpfit_result) << "]." << std::endl;
    std::cout << "Initial chi-square: " << mpfit_result_info.orignorm << std::endl;
    std::cout << "Final chi-square: " << mpfit_result_info.bestnorm << std::endl;

    //
    // Extract fitted model parameters.
    //

    std::cout << "Fitting results:" << std::endl;
    for (std::size_t i = 0; i < param_names.size(); ++i)
    {
        std::string param_name = param_names[i];
        float param_best = param_values[i];
        float param_stddev = mpfit_result_info.xerror[i];

        std::cout << std::fixed
                  << std::setprecision(2)
                  << "Fitted param:"
                  << " name="   << std::setw(4) << param_name   << ","
                  << " fixed="  << std::setw(2) << mpfit_params_info[i].fixed   << ","
                  << " best="   << std::setw(8) << param_best   << ","
                  << " stddev=" << std::setw(8) << param_stddev << std::endl;

        params_info.get_parameter(param_name)
                .add("best", param_best)
                .add("stddev", param_stddev);
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

} // namepsece mpfit
} // namespace fitters
} // namespace gbkfit
