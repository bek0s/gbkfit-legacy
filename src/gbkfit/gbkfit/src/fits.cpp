
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
    else if (std::is_same<T, unsigned short>::value)    // ???
        type = USHORT_IMG;
    else if (std::is_same<T, short>::value)
        type = SHORT_IMG;
    else if (std::is_same<T, unsigned long>::value)     // ???
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

ndarray* get_data(const std::string& filename)
{
    fitsfile* fp = nullptr;
    int status = 0;
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

    // Get fits data type. For now convert everything to float.
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

    fitsfile* fp = nullptr;
    int status = 0;
    int naxis = 0;
    long naxes[512];
    int bitpix = 0;
    int datatype = 0;
    long firstpix[512];
    long long nelem = 0;

    // Get image dimension count.
    naxis = data.get_shape().get_dim_count();

    // Get image dimension length.
    std::copy(shape.get_as_vector().begin(),shape.get_as_vector().end(),naxes);

    // Select image type.
    bitpix = select_fits_image_type<float>();

    // Select pixel data type. For now convert everything to float.
    datatype = select_fits_data_type<float>();

    // Set the first pixel for each dimension (indices start from 1).
    std::fill_n(firstpix,naxis,1);

    // Get number of pixels.
    nelem = shape.get_dim_length_product();

    // Create a copy of the data on the host. We use std::unique_pointer as an exception guard.
    std::unique_ptr<ndarray_host> data_host = std::make_unique<ndarray_host_new>(data);

    // Create new file.
    fits_create_file(&fp,filename.c_str(),&status);

    // Create an image in the new file.
    fits_create_img(fp,bitpix,naxis,naxes,&status);

    // Write pixels to the image.
    fits_write_pix(fp,datatype,firstpix,nelem,data_host->get_host_ptr(),&status);

    // Close fits file.
    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
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
