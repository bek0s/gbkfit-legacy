
#include <CCfits/CCfits>
#include "gbkfit/fits_util.hpp"

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


void get_header(const std::string& name)
{


    int status = 0;
    fitsfile* file;

    std::cout << "file: " << name << std::endl;

    fits_open_file(&file,name.c_str(),READONLY,&status);

    int card_count;

    fits_get_hdrspace(file,&card_count,nullptr,&status);

    std::cout << "n: " << card_count << std::endl;

    for(int i = 0; i < card_count; ++i)
    {

    }


    std::complex<float> foo;
    foo.real(4);
    foo.imag(6);

    std::cout << foo << std::endl;

    float foo2 = 10;

//  fits_write_key(file,TCOMPLEX,"NEW2",&foo,nullptr,&status);

    std::cout << "status: " << status << std::endl;

//  fits_delete_key(file,"HISTORY",&status);

//  fits_delete_str(file,"comment",&status);

    char foo3[80];
    fits_read_key(file,TSTRING,"COMMENT",foo3,nullptr,&status);

    std::cout << "status (2): " << status << "--" << get_errstatus_string(status) << std::endl;

    fits_close_file(file,&status);



}

ndarray* get_data(const std::string& name)
{
    std::shared_ptr<CCfits::FITS> fits = std::make_shared<CCfits::FITS>(name,CCfits::Read,true);
    CCfits::PHDU& fits_img = fits->pHDU();

    std::valarray<float> valarray;
    fits_img.read<float>(valarray);

    double vv;
    fits->pHDU().readKey("NAXIS",vv);

    std::vector<std::string> names;
    std::vector<long> values;
    auto foo = fits->pHDU().keyWord();


    foo["NAXIS"]->value<double>(vv);

    std::cout << vv << std::endl;



    std::vector<size_t> axes;
    for(long int i = 0; i < fits_img.axes(); ++i)
        axes.push_back(static_cast<size_t>(fits_img.axis(i)));

    return new ndarray_host(ndshape(axes),std::begin(valarray));
}

void write_to(const std::string& name, const ndarray& data)
{
    long naxis = data.get_shape().get_dim_count();
    std::vector<long> naxes;
    for(std::size_t i = 0; i < static_cast<std::size_t>(naxis); ++i)
        naxes.push_back(data.get_shape().get_dim_length(i));

    std::shared_ptr<CCfits::FITS> fits = std::make_shared<CCfits::FITS>(name,FLOAT_IMG,naxis,naxes.data());


    auto length = data.get_shape().get_dim_length_product();
    float* data_raw = new float[length];

    data.read_data(data_raw);

    std::valarray<float> data_val(data_raw,length);


    long fpixel(1);
    fits->pHDU().write(fpixel,length,data_val);

}

void write_to2(const std::string& name, const ndarray& data)
{
    long naxis = data.get_shape().get_dim_count();
    std::vector<long> naxes;
    for(std::size_t i = 0; i < static_cast<std::size_t>(naxis); ++i)
        naxes.push_back(data.get_shape().get_dim_length(i));

    std::shared_ptr<CCfits::FITS> fits = std::make_shared<CCfits::FITS>(name,FLOAT_IMG,naxis,naxes.data());

    long fpixel(1);
//  fits->pHDU().write(fpixel,data.get_data().size(),data.get_data());


}

ndarray* get_data2(const std::string& filename)
{
    int status = 0;
    fitsfile* fp;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READONLY,&status);

}

template<typename T>
void get_keyword(const std::string& filename, const std::string& keyname, T& value, std::string& comment)
{
    int status = 0;
    fitsfile* fp;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    status = 0;
    fits_read_key(fp,0,keyname.c_str(),value,nullptr);

    status = 0;
    fits_close_file(fp,&status);
}

template<typename T>
void get_keyword_value(const std::string& filename, const std::string& keyname, T& value)
{
    int status = 0;
    fitsfile* fp;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    status = 0;
    fits_read_key(fp,0,keyname.c_str(),value,nullptr);

    status = 0;
    fits_close_file(fp,&status);
}

void get_keyword_comment(const std::string& filename, const std::string& keyname, std::string& comment)
{
    int status = 0;
    fitsfile* fp;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READONLY,&status);

    status = 0;
    fits_read_keyword(fp,nullptr,nullptr,nullptr,&status);

    status = 0;
    fits_close_file(fp,&status);
}

template<typename T>
void set_keyword(const std::string& filename, const std::string& keyname, const T& value, const std::string& comment)
{
    int status = 0;
    fitsfile* fp;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READWRITE,&status);

    status = 0;
    fits_update_key(fp,0,keyname.c_str(),value,comment.c_str(),&status);

    status = 0;
    fits_close_file(fp,&status);
}

template<typename T>
void set_keyword_value(const std::string& filename, const std::string& keyname, const T& value)
{
    int status = 0;
    fitsfile* fp;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READWRITE,&status);

    status = 0;
    fits_update_key(fp,0,keyname.c_str(),value,nullptr,&status);

    status = 0;
    fits_close_file(fp,&status);
}

void set_keyword_comment(const std::string& filename, const std::string& keyname, const std::string& comment)
{
    int status = 0;
    fitsfile* fp;

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
    fitsfile* fp;

    status = 0;
    fits_open_file(&fp,filename.c_str(),READWRITE,&status);

    status = 0;
    fits_delete_key(fp,keyname.c_str(),&status);

    status = 0;
    fits_close_file(fp,&status);
}


} // namespace fits_util
} // namespace gbkfit
