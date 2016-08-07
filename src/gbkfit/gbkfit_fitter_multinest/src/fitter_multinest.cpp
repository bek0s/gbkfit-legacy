
#include "gbkfit/fitter/multinest/fitter_multinest.hpp"
#include "gbkfit/fitter/multinest/fitter_multinest_factory.hpp"

#include "gbkfit/dataset.hpp"
#include "gbkfit/dmodel.hpp"
#include "gbkfit/gmodel.hpp"
#include "gbkfit/ndarray_host.hpp"
#include "gbkfit/params.hpp"
#include "gbkfit/utility.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <multinest.h>

#include "gbkfit/fitter_result.hpp"

namespace gbkfit {
namespace fitter {
namespace multinest {

struct multinest_user_data
{
    const gbkfit::DModel* dmodel;
    std::vector<std::string> param_name;
    std::vector<bool> param_fixed;
    std::vector<float> param_value;
    std::vector<float> param_min;
    std::vector<float> param_max;
    std::vector<float> param_mean;
    std::vector<float> param_stddev;
    std::vector<float> param_best;
    std::vector<float> param_map;
    std::vector<std::string> dataset_names;
    std::map<std::string,NDArrayHost*> data_map_dat;
    std::map<std::string,NDArrayHost*> data_map_msk;
    std::map<std::string,NDArrayHost*> data_map_err;
    std::map<std::string,NDArrayHost*> data_map_mdl;
}; // struct multinest_user_data

void multinest_callback_likelihood(double* cube, int& ndim, int& npars, double& lnew, void* context)
{
    (void)ndim;
    (void)npars;

    //
    // Get a convenience pointer to the user data.
    //

    multinest_user_data* udata = reinterpret_cast<multinest_user_data*>(context);

    //
    //  Update model parameter values.
    //

    int free_params_counter = 0;
    for(std::size_t i = 0; i < udata->param_name.size(); ++i)
    {
        if(!udata->param_fixed[i])
        {
            udata->param_value[i] = udata->param_min[i] + (udata->param_max[i] - udata->param_min[i]) * cube[free_params_counter];
            cube[free_params_counter] = udata->param_value[i];
            free_params_counter++;
        }
    }

    //
    // Build parameter map with the current set of parameter values.
    //

    const std::map<std::string,float> param_map = gbkfit::vectors_to_map(udata->param_name, udata->param_value);

#if 0
    // Provide debug information about the current set of parameters.
    std::cout << gbkfit::to_string(param_map) << std::endl;
#endif

    //
    // Evaluate model with the current set of parameter values.
    //

    std::map<std::string,NDArrayHost*> model_data = udata->dmodel->evaluate(param_map);

    //
    // Calculate likelihood between the model and the (available) data.
    //

    float chi2 = 0;
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

        // Calculate chi squared.
        for(int i = 0; i < data_size; ++i) {
            float residual = data_msk[i] > 0 ? (data_dat[i]-data_mdl[i])/data_err[i] : 0;
            chi2 += residual*residual;
        }
    }

//  std::cout << chi2 << std::endl;

    // Save likelihood! Woohoo!
    lnew = -0.5*chi2;
}

void multinest_callback_dumper(int& nsamples, int& nlive, int& npar, double** physlive, double** posterior, double** paramconstr, double& maxloglike, double& logz, double& inslogz, double& logzerr, void* context)
{
    (void)nsamples;
    (void)nlive;
    (void)physlive;
    (void)posterior;
    (void)maxloglike;
    (void)logz;
    (void)inslogz;
    (void)logzerr;

    multinest_user_data* udata = reinterpret_cast<multinest_user_data*>(context);

    const std::map<std::string,float> param_map = gbkfit::vectors_to_map(udata->param_name, udata->param_value);


    // Provide debug information about the current set of parameters.
//  std::cout << gbkfit::to_string(param_map) << std::endl;

    std::stringstream str;

    int free_params_counter = 0;
    for(std::size_t i = 0; i < udata->param_name.size(); ++i)
    {


        if(!udata->param_fixed[i])
        {
            udata->param_mean[i]   = paramconstr[0][npar*0+free_params_counter];
            udata->param_stddev[i] = paramconstr[0][npar*1+free_params_counter];
            udata->param_best[i]   = paramconstr[0][npar*2+free_params_counter];
            udata->param_map[i]    = paramconstr[0][npar*3+free_params_counter];
            free_params_counter++;
        }

        str << udata->param_name[i] << ": " << udata->param_mean[i] << ", ";
    }

    std::cout << str.str() << std::endl;

}

const double FitterMultinest::DEFAULT_EFR = 1.0;
const double FitterMultinest::DEFAULT_TOL = 0.5;
const double FitterMultinest::DEFAULT_ZTOL = -1e90;
const double FitterMultinest::DEFAULT_LOGZERO = -1e90;
const int FitterMultinest::DEFAULT_IS = 1;
const int FitterMultinest::DEFAULT_MMODAL = 0;
const int FitterMultinest::DEFAULT_CEFF = 0;
const int FitterMultinest::DEFAULT_NLIVE = 50;
const int FitterMultinest::DEFAULT_MAXITER = 2000;
const int FitterMultinest::DEFAULT_SEED = 1;

FitterMultinest::FitterMultinest(void)
    : m_efr(DEFAULT_EFR)
    , m_tol(DEFAULT_TOL)
    , m_ztol(DEFAULT_ZTOL)
    , m_logzero(DEFAULT_LOGZERO)
    , m_is(DEFAULT_IS)
    , m_mmodal(DEFAULT_MMODAL)
    , m_ceff(DEFAULT_CEFF)
    , m_nlive(DEFAULT_NLIVE)
    , m_maxiter(DEFAULT_MAXITER)
    , m_seed(DEFAULT_SEED)
{
}

FitterMultinest::~FitterMultinest()
{
}

const std::string& FitterMultinest::get_type(void) const
{
    return FitterMultinestFactory::FACTORY_TYPE;
}

FitterResult* FitterMultinest::fit(const DModel* dmodel, const Params* params, const std::vector<Dataset*>& datasets) const
{
    //
    // Perform the necessary memory allocations and copy the datasets on the host.
    //

    std::vector<std::string> dataset_names;
    std::map<std::string,NDArrayHost*> data_map_dat;
    std::map<std::string,NDArrayHost*> data_map_msk;
    std::map<std::string,NDArrayHost*> data_map_err;
    std::map<std::string,NDArrayHost*> data_map_mdl;

    // Iterate over the input datasets.
    for(auto& dataset : datasets)
    {
        // Get the dataset name and store it in a vector, we will use it later for convenience.
        std::string dataset_name = dataset->get_name();
        dataset_names.push_back(dataset_name);

        // Get input data pointer shortcuts for convenience.
        const NDArray* input_data_dat = dataset->get_data();
        const NDArray* input_data_msk = dataset->get_mask();
        const NDArray* input_data_err = dataset->get_errors();

        // Allocate host memory for the input data and the model.
        data_map_dat[dataset_name] = new NDArrayHost(input_data_dat->get_shape());
        data_map_msk[dataset_name] = new NDArrayHost(input_data_msk->get_shape());
        data_map_err[dataset_name] = new NDArrayHost(input_data_err->get_shape());
        data_map_mdl[dataset_name] = new NDArrayHost(input_data_dat->get_shape());

        // Copy data to the host.
        data_map_dat[dataset_name]->write_data(input_data_dat);
        data_map_msk[dataset_name]->write_data(input_data_msk);
        data_map_err[dataset_name]->write_data(input_data_err);
    }

    //
    // Get model parameters. Always use this order for the parameters!
    //

    std::vector<std::string> param_names = dmodel->get_galaxy_model()->get_param_names();

    //
    //  build arrays with the free parameters only
    //


    multinest_user_data udata;

    udata.param_fixed.resize(param_names.size());
    udata.param_value.resize(param_names.size());
    udata.param_min.resize(param_names.size());
    udata.param_max.resize(param_names.size());
    udata.param_mean.resize(param_names.size());
    udata.param_stddev.resize(param_names.size());
    udata.param_best.resize(param_names.size());
    udata.param_map.resize(param_names.size());

    int free_param_counter = 0;
    for(std::size_t i = 0; i < param_names.size(); ++i)
    {
        std::string param_name = param_names[i];

        udata.param_fixed[i] = params->get(param_name).get<bool>("fixed",0);

        if(!udata.param_fixed[i])
        {
            udata.param_min[i] = params->get(param_name).get<float>("min");
            udata.param_max[i] = params->get(param_name).get<float>("max");
            free_param_counter++;
        }
        else
        {
            udata.param_value[i] = params->get(param_name).get<float>("value");
            udata.param_mean[i] = udata.param_value[i];
            udata.param_stddev[i] = 0.0;
            udata.param_best[i] = udata.param_value[i];
            udata.param_map[i]= udata.param_value[i];
        }
    }

    udata.dmodel = dmodel;
    udata.param_name = param_names;
    udata.dataset_names = dataset_names;
    udata.data_map_dat = data_map_dat;
    udata.data_map_msk = data_map_msk;
    udata.data_map_err = data_map_err;
    udata.data_map_mdl = data_map_mdl;

    //
    // ...
    //

    int IS = m_is;                         // do Nested Importance Sampling?
    int mmodal = m_mmodal;                     // do mode separation?
    int ceff = m_ceff;                       // run in constant efficiency mode?
    int nlive = m_nlive;                // number of live points
    double efr = m_efr;                   // set the required efficiency
    double tol = m_tol;                   // tol, defines the stopping criteria
    int ndims = free_param_counter;     // dimensionality (no. of free parameters)
    int nPar = free_param_counter;      // total no. of parameters including free & derived parameters
    int nClsPar = free_param_counter;   // no. of parameters to do mode separation on
    int updInt = 1;                   // after how many iterations feedback is required & the output files should be updated
    double Ztol = -3.40282e+37;                // all the modes with logZ < Ztol are ignored
    int maxModes = 20;                 // expected max no. of modes (used only for memory allocation)
    int* pWrap = new int[ndims];        // which parameters to have periodic boundary conditions?
    for(int i = 0; i < ndims; i++)
        pWrap[i] = 0;
    char root[100] = "multinest_";             // root for output files
    int seed = m_seed;                  // random no. generator seed, if < 0 then take the seed from system clock
    int fb = 0;                         // need feedback on standard output?
    int resume = 0;                     // resume from a previous job?
    int outfile = 1;                    // write output files?
    int initMPI = 0;                    // initialize MPI routines?, relevant only if compiling with MPI
                                        // set it to F if you want your main program to handle MPI initialization
    double logZero = -3.40282e+37;             // points with loglike < logZero will be ignored by MultiNest
    int maxiter = m_maxiter;            // max no. of iterations, a non-positive value means infinity. MultiNest will terminate if either it
                                        // has done max no. of iterations or convergence criterion (defined through tol) has been satisfied

    nested::run(IS,
                mmodal,
                ceff,
                nlive,
                tol,
                efr,
                ndims,
                nPar,
                nClsPar,
                maxModes,
                updInt,
                Ztol,
                root,
                seed,
                pWrap,
                fb,
                resume,
                outfile,
                initMPI,
                logZero,
                maxiter,
                multinest_callback_likelihood,
                multinest_callback_dumper,
                &udata);

    //
    // Extract fitted model parameters.
    //

    std::vector<bool> param_fixed_flags;
    std::vector<std::vector<float>> param_values_list;
    std::vector<std::vector<float>> param_errors_list;

    for(std::size_t i = 0; i < param_names.size(); ++i)
    {
        param_fixed_flags.push_back(udata.param_fixed[i]);
    }

    for(std::size_t i = 0; i < 1; ++i)
    {
        std::vector<float> param_values;
        std::vector<float> param_errors;

        for(std::size_t j = 0; j < param_names.size(); ++j)
        {
            param_values.push_back(udata.param_best[j]);
            param_errors.push_back(udata.param_stddev[j]);
        }

        param_values_list.push_back(param_values);
        param_errors_list.push_back(param_errors);
    }




    std::vector<NDArray*> dataset_data;
    std::vector<NDArray*> dataset_errors;
    std::vector<NDArray*> dataset_masks;

    for(auto& dataset : datasets)
    {
        dataset_data.push_back(dataset->get_data());
        dataset_errors.push_back(dataset->get_errors());
        dataset_masks.push_back(dataset->get_mask());
    }

    FitterResult* result = new FitterResult(dmodel,
                                            dataset_names,
                                            dataset_data,
                                            dataset_errors,
                                            dataset_masks,
                                            param_fixed_flags,
                                            param_values_list,
                                            param_errors_list);




    //
    // Perform clean up
    //

    delete [] pWrap;
    for(auto& data : data_map_mdl) delete std::get<1>(data);
    for(auto& data : data_map_dat) delete std::get<1>(data);
    for(auto& data : data_map_msk) delete std::get<1>(data);
    for(auto& data : data_map_err) delete std::get<1>(data);

    //FitterResult* result = new FitterResult(fitter, model, data);

    //result->add_mode(parameters, model, residual);





    return result;
}

} // namespace multinest
} // namespace fitters
} // namespace gbkfit
