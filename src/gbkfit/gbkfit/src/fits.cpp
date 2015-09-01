
#include "gbkfit/fits.hpp"
#include "gbkfit/ndarray_host.hpp"

#include <fitsio2.h>

namespace gbkfit {
namespace fits {

//!
//! \brief select_fits_data_type
//! \return
//!
template<typename T>
int select_fits_data_type(void)
{
    int type = 0;
    if      (std::is_same<T, bool>::value)
        type = TLOGICAL;
    else if (std::is_same<T, unsigned char>::value)
        type = TBYTE;
    else if (std::is_same<T, char>::value)
        type = TSBYTE;
    else if (std::is_same<T, unsigned short>::value)
        type = TUSHORT;
    else if (std::is_same<T, short>::value)
        type = TSHORT;
    else if (std::is_same<T, unsigned int>::value)
        type = TUINT;
    else if (std::is_same<T, unsigned>::value)
        type = TUINT;
    else if (std::is_same<T, unsigned long>::value)
        type = TULONG;
    else if (std::is_same<T, long>::value)
        type = TLONG;
    else if (std::is_same<T, long long>::value)
        type = TLONGLONG;
    else if (std::is_same<T, float>::value)
        type = TFLOAT;
    else if (std::is_same<T, double>::value)
        type = TDOUBLE;
    else if (std::is_same<T, std::complex<float>>::value)
        type = TCOMPLEX;
    else if (std::is_same<T, std::complex<double>>::value)
        type = TDBLCOMPLEX;
    else
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    return type;
}

//!
//! \brief select_fits_image_type
//! \return
//!
template<typename T>
int select_fits_image_type(void)
{
    int type = 0;
    if      (std::is_same<T, unsigned char>::value)
        type = BYTE_IMG;
    else if (std::is_same<T, char>::value)
        type = SBYTE_IMG;
    else if (std::is_same<T, unsigned short>::value)
        type = USHORT_IMG;
    else if (std::is_same<T, short>::value)
        type = SHORT_IMG;
    else if (std::is_same<T, unsigned long>::value)
        type = ULONG_IMG;
    else if (std::is_same<T, long>::value)
        type = LONG_IMG;
    else if (std::is_same<T, long long>::value)
        type = LONGLONG_IMG;
    else if (std::is_same<T, float>::value)
        type = FLOAT_IMG;
    else if (std::is_same<T, double>::value)
        type = DOUBLE_IMG;
    else
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    return type;
}

header get_header(const std::string& filename)
{
    int status = 0;
    fitsfile* fp = nullptr;

    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    int nexist = 0;
    int nmore = 0;
    fits_get_hdrspace(fp,&nexist,&nmore,&status);

    // TODO

    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return header();
}

ndarray* get_data(const std::string& filename)
{
    int status = 0;
    fitsfile* fp = nullptr;
    int naxis = 0;
    long naxes[512];
    int datatype = 0;
    long firstpix[512];
    long long nelem = 0;
    int anynul = 0;
    void* nulval = nullptr;

    // Open fits file.
    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    // Get image dimension count.
    fits_get_img_dim(fp,&naxis,&status);

    // Get image dimension length.
    fits_get_img_size(fp,naxis,naxes,&status);

    // Create shape
    ndshape shape(std::vector<ndshape::value_type>(naxes,naxes+naxis));

    // Allocate data. We use std::unique_pointer as an exception guard.
    std::unique_ptr<ndarray_host> data = std::make_unique<ndarray_host_new>(shape);

    // Select fits data type. For now convert everything to float.
    datatype = select_fits_data_type<float>();

    // Set the first pixel for each dimension (indices start from 1).
    std::fill_n(firstpix,naxis,1);

    // Get number of pixels.
    nelem = shape.get_dim_length_product();

    // Read pixels.
    fits_read_pix(fp,datatype,firstpix,nelem,nulval,data->get_host_ptr(),&anynul,&status);

    // Close fits file.
    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return data.release();
}

void write_to(const std::string& filename, const ndarray& data)
{
    const ndshape& shape = data.get_shape();

    int status = 0;
    fitsfile* fp = nullptr;
    int naxis = shape.get_dim_count();
    long naxes[512];
    int bitpix = select_fits_image_type<float>();
    int datatype = select_fits_data_type<float>();
    long firstpix[512];
    long long nelem = shape.get_dim_length_product();

    // Get image dimension length.
    std::copy(shape.get_as_vector().begin(),shape.get_as_vector().end(),naxes);

    // Set the first pixel for each dimension (indices start from 1).
    std::fill_n(firstpix,naxis,1);

    // Create a copy of the data on the host. We use std::unique_pointer as an exception guard.
    std::unique_ptr<ndarray_host> data_host = std::make_unique<ndarray_host_new>(data);

    fits_create_file(&fp,filename.c_str(),&status);

    fits_create_img(fp,bitpix,naxis,naxes,&status);

    fits_write_pix(fp,datatype,firstpix,nelem,data_host->get_host_ptr(),&status);

    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

template<typename T>
void get_key(const std::string& filename, const std::string& keyname, T& value, std::string& comment)
{
    get_key_value<T>(filename,keyname,value);
    get_key_comment(filename,keyname,comment);
}

template<typename T>
void get_key_value(const std::string& filename, const std::string& keyname, T& value)
{
    int status = 0;
    fitsfile* fp = nullptr;
    int datatype = select_fits_data_type<T>();

    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    fits_read_key(fp,datatype,keyname.c_str(),&value,nullptr,&status);

    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

void get_key_comment(const std::string& filename, const std::string& keyname, std::string& comment)
{
    int status = 0;
    fitsfile* fp = nullptr;
    char keyval[512];
    char keycomm[512];

    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    fits_read_keyword(fp,keyname.c_str(),keyval,keycomm,&status);
    comment = std::string(keycomm);

    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

template<typename T>
void set_key(const std::string& filename, const std::string& keyname, const T& value, const std::string& comment)
{
    set_key_value<T>(filename,keyname,value);
    set_key_comment(filename,keyname,comment);
}

template<typename T>
void set_key_value(const std::string& filename, const std::string& keyname, const T& value)
{
    int status = 0;
    fitsfile* fp = nullptr;
    int datatype = select_fits_data_type<T>();

    fits_open_file(&fp,filename.c_str(),READWRITE,&status);
    
    fits_update_key(fp,datatype,keyname.c_str(),const_cast<T*>(&value),nullptr,&status);

    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

void set_key_comment(const std::string& filename, const std::string& keyname, const std::string& comment)
{
    int status = 0;
    fitsfile* fp = nullptr;

    fits_open_file(&fp,filename.c_str(),READWRITE,&status);

    fits_modify_comment(fp,keyname.c_str(),comment.c_str(),&status);

    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

void del_key(const std::string& filename, const std::string& keyname)
{
    int status = 0;
    fitsfile* fp = nullptr;

    fits_open_file(&fp,filename.c_str(),READWRITE,&status);

    fits_delete_key(fp,keyname.c_str(),&status);

    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

template void get_key<float>(const std::string& filename, const std::string& keyname, float& value, std::string& comment);
template void get_key_value<float>(const std::string& filename, const std::string& keyname, float& value);
template void set_key<float>(const std::string& filename, const std::string& keyname, const float& value, const std::string& comment);
template void set_key_value<float>(const std::string& filename, const std::string& keyname, const float& value);

} // namespace fits
} // namespace gbkfit
