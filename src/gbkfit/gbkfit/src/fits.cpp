
#include <CCfits/CCfits>
#include "gbkfit/fits.hpp"

#include "gbkfit/ndarray_host.hpp"

#include <algorithm>
#include <functional>

namespace gbkfit {
namespace fits_util {

std::string get_errstatus_string(int status)
{
    char errtext[80];
    fits_get_errstatus(status,errtext);
    return std::string(errtext);
}


}

namespace fits {


template <typename T> int select_fits_data_type();
template <> int select_fits_data_type<bool>(void) { return TLOGICAL; }
template <> int select_fits_data_type<unsigned char>(void) { return TBYTE; }
template <> int select_fits_data_type<char>(void) { return TSBYTE; }
template <> int select_fits_data_type<unsigned short>(void) { return TUSHORT; }
template <> int select_fits_data_type<short>(void) { return TSHORT; }
template <> int select_fits_data_type<unsigned int>(void) { return TUINT; }
template <> int select_fits_data_type<int>(void) { return TINT; }
template <> int select_fits_data_type<unsigned long>(void) { return TULONG; }
template <> int select_fits_data_type<long>(void) { return TLONG; }
template <> int select_fits_data_type<long long>(void) { return TLONGLONG; }
template <> int select_fits_data_type<float>(void) { return TFLOAT; }
template <> int select_fits_data_type<double>(void) { return TDOUBLE; }
template <> int select_fits_data_type<std::complex<float>>(void) { return TCOMPLEX; }
template <> int select_fits_data_type<std::complex<double>>(void) { return TDBLCOMPLEX; }

ndarray* get_data(const std::string& filename)
{
    int status = 0;
    fitsfile* fp = nullptr;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    int naxis;
    status = 0;
    fits_get_img_dim(fp,&naxis,&status);


    long int naxes[512];
    status = 0;
    fits_get_img_size(fp,naxis,naxes,&status);

    ndshape shape(std::vector<ndshape::value_type>(naxes,naxes+naxis));
    ndarray_host* array = new ndarray_host_new(shape);

    int datatype = select_fits_data_type<float>();
    long int firstpix[512] = {1};
    long long int nelem = shape.get_dim_length_product();
    int anynul = 0;
    status = 0;
    fits_read_pix(fp,datatype,firstpix,nelem,nullptr,array->get_host_ptr(),&anynul,&status);

    status = 0;
    fits_close_file(fp,&status);

    return array;
}

void write_to(const std::string& filename, const ndarray& data)
{

    long naxis = data.get_shape().get_dim_count();
    std::vector<long> naxes;
    for(std::size_t i = 0; i < static_cast<std::size_t>(naxis); ++i)
        naxes.push_back(data.get_shape().get_dim_length(i));

    std::shared_ptr<CCfits::FITS> fits = std::make_shared<CCfits::FITS>(filename,FLOAT_IMG,naxis,naxes.data());


    auto length = data.get_shape().get_dim_length_product();
    float* data_raw = new float[length];

    data.read_data(data_raw);

    std::valarray<float> data_val(data_raw,length);


    long fpixel(1);
    fits->pHDU().write(fpixel,length,data_val);


    /*
    int status = 0;
    fitsfile* fp = nullptr;

    status = 0;
    fits_create_file(&fp,filename.c_str(),&status);

    int datatype = select_fits_data_type<float>();
    long int firstpix[512] = {1};
    long long int nelem = data.get_shape().get_dim_length_product();


    float* data_raw = new float[nelem];
    data.read_data(data_raw);

    fits_write_pix(fp,datatype,firstpix,nelem,data_raw,&status);

    delete [] data_raw;

    status = 0;
    fits_close_file(fp,&status);

    */


}


template<typename T>
void get_keyword(const std::string& filename, const std::string& keyname, T& value, std::string& comment)
{
    get_keyword_value<T>(filename,keyname,value);
    get_keyword_comment(filename,keyname,comment);
}

template<typename T>
void get_keyword_value(const std::string& filename, const std::string& keyname, T& value)
{
    int status = 0;
    fitsfile* fp = nullptr;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    int datatype = select_fits_data_type<T>();
    char* comm = nullptr;
    status = 0;
    fits_read_key(fp,datatype,keyname.c_str(),&value,comm,&status);

    status = 0;
    fits_close_file(fp,&status);
}

void get_keyword_comment(const std::string& filename, const std::string& keyname, std::string& comment)
{
    int status = 0;
    fitsfile* fp = nullptr;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    char keyval[512];
    char comm[512];
    status = 0;
    fits_read_keyword(fp,keyname.c_str(),keyval,comm,&status);
    comment = std::string(comm);

    status = 0;
    fits_close_file(fp,&status);
}

template<typename T>
void set_keyword(const std::string& filename, const std::string& keyname, const T& value, const std::string& comment)
{
    set_keyword_value<T>(filename,keyname,value);
    set_keyword_comment(filename,keyname,comment);
}

template<typename T>
void set_keyword_value(const std::string& filename, const std::string& keyname, const T& value)
{
    int status = 0;
    fitsfile* fp = nullptr;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READWRITE,&status);

    int datatype = select_fits_data_type<T>();
    status = 0;
    fits_update_key(fp,datatype,keyname.c_str(),const_cast<T*>(&value),nullptr,&status);

    status = 0;
    fits_close_file(fp,&status);
}

void set_keyword_comment(const std::string& filename, const std::string& keyname, const std::string& comment)
{
    int status = 0;
    fitsfile* fp = nullptr;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READWRITE,&status);

    status = 0;
    fits_modify_comment(fp,keyname.c_str(),comment.c_str(),&status);

    status = 0;
    fits_close_file(fp,&status);
}

void del_keyword(const std::string& filename, const std::string& keyname)
{
    int status = 0;
    fitsfile* fp = nullptr;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READWRITE,&status);

    status = 0;
    fits_delete_key(fp,keyname.c_str(),&status);

    status = 0;
    fits_close_file(fp,&status);
}

template void get_keyword<float>(const std::string& filename, const std::string& keyname, float& value, std::string& comment);
template void get_keyword_value<float>(const std::string& filename, const std::string& keyname, float& value);
template void set_keyword<float>(const std::string& filename, const std::string& keyname, const float& value, const std::string& comment);
template void set_keyword_value<float>(const std::string& filename, const std::string& keyname, const float& value);


} // namespace fits
} // namespace gbkfit
