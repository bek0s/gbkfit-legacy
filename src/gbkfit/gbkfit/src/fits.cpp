
#include "gbkfit/fits.hpp"
#include "gbkfit/ndarray_host.hpp"

#include <boost/lexical_cast.hpp>
#include <fitsio.h>

namespace gbkfit {
namespace fits {

template<typename T>
int select_fits_data_type(void)
{
    int type = 0;
    if      (std::is_same<T, bool>::value)
        type = TLOGICAL;
    else if (std::is_same<T, char>::value)
        type = TSBYTE;
    else if (std::is_same<T, unsigned char>::value)
        type = TBYTE;
    else if (std::is_same<T, short>::value)
        type = TSHORT;
    else if (std::is_same<T, unsigned short>::value)
        type = TUSHORT;
    else if (std::is_same<T, int>::value)
        type = TINT;
    else if (std::is_same<T, unsigned int>::value)
        type = TUINT;
    else if (std::is_same<T, long>::value)
        type = TLONG;
    else if (std::is_same<T, unsigned long>::value)
        type = TULONG;
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
    else if (std::is_same<T, std::string>::value)
        type = TSTRING;
    else
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    return type;
}

template<typename T>
int select_fits_image_type(void)
{
    int type = 0;
    if      (std::is_same<T, unsigned char>::value)
        type = BYTE_IMG;
    else if (std::is_same<T, short>::value)
        type = SHORT_IMG;
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

std::string get_status_message(int status)
{
    char message[FLEN_STATUS];
    fits_get_errstatus(status, message);
    return std::string(message);
}

std::vector<std::string> get_error_message_stack(void)
{
    std::vector<std::string> messages;
    char message[FLEN_ERRMSG];
    while(fits_read_errmsg(message))
        messages.push_back(std::string(message));
    return messages;
}

Header get_header(const std::string& filename)
{
    int status = 0;
    fitsfile* fp = nullptr;
    fits_open_file(&fp, filename.c_str(), READONLY, &status);

    int nexist = 0;
    int nmore = 0;
    fits_get_hdrspace(fp, &nexist, &nmore, &status);

    Header header;

    for(int i = 0; i < nexist; ++i)
    {
        char key[FLEN_CARD];
        char value[FLEN_VALUE];
        char comment[FLEN_COMMENT];
        fits_read_keyn(fp, i+1, key, value, comment, &status);

        if      (std::string(key) == "COMMENT")
        {
            // Comment keyword
            // TODO
        }
        else if (std::string(key) == "HISTORY")
        {
            // History keyword
            // TODO
        }
        else if (std::string(key) == "CONTINUE")
        {
            // Continuation of long string value
            // TODO
        }
        else
        {
            Header::Keyword& keyword = header.add_keyword(key);

            if (strlen(value) > 0)
            {
                char dtype;
                fits_get_keytype(value, &dtype, &status);

                if      (dtype == 'L')
                {
                    int value_;
                    fits_read_key_log(fp, key, &value_, nullptr, &status);
                    keyword.set<bool>(value_, comment);
                }
                else if (dtype == 'I')
                {
                    long long value_;
                    fits_read_key_lnglng(fp, key, &value_, nullptr, &status);
                    keyword.set<long long>(value_, comment);
                }
                else if (dtype == 'F')
                {
                    double value_;
                    fits_read_key_dbl(fp, key, &value_, nullptr, &status);
                    keyword.set<double>(value_, comment);
                }
                else if (dtype == 'X')
                {
                    std::complex<double> value_;
                    fits_read_key_dblcmp(fp, key, reinterpret_cast<double*>(&value_), nullptr, &status);
                    keyword.set<std::complex<double>>(value_, comment);
                }
                else if (dtype == 'C')
                {
                    char* value_;
                    fits_read_key_longstr(fp, key, &value_, nullptr, &status);
                    keyword.set<std::string>(value_, comment);
                    fits_free_memory(value_, &status);
                }
            }
            else
            {
                // Keyword with null value
                // TODO (?)
            }
        }
    }

    fits_close_file(fp,&status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return header;
}

std::unique_ptr<NDArrayHost> get_data(const std::string& filename)
{
    int status = 0;
    fitsfile* fp = nullptr;
    int naxis = 0;
    long naxes[512];
    int datatype = select_fits_data_type<float>();
    long firstpix[512];
    long long nelem = 0;
    int anynul = 0;
    void* nulval = nullptr;

    fits_open_file(&fp, filename.c_str(), READONLY, &status);
    fits_get_img_dim(fp, &naxis, &status);
    fits_get_img_size(fp, naxis, naxes, &status);

    std::unique_ptr<NDArrayHost> data = std::make_unique<NDArrayHost>(NDShape(naxes, naxes+naxis));
    std::fill_n(firstpix, naxis, 1);
    nelem = static_cast<long long>(data->get_size());

    fits_read_pix(fp, datatype, firstpix, nelem, nulval, data->get_host_ptr(), &anynul, &status);
    fits_close_file(fp, &status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    return data;
}

void write_to(const std::string& filename, const NDArray& data)
{
    const NDShape& shape = data.get_shape();

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
    std::unique_ptr<NDArrayHost> data_host = std::make_unique<NDArrayHost>(shape);
    //data_host->write_data(&data);

    data.read_data(data_host->get_host_ptr());

    fits_create_file(&fp, filename.c_str(), &status);
    fits_create_img(fp, bitpix, naxis, naxes, &status);
    fits_write_pix(fp, datatype, firstpix, nelem, data_host->get_host_ptr(), &status);
    fits_close_file(fp, &status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

Header::Keyword::Keyword(const std::string& key)
    : m_key(key)
    , m_datatype(0)
{
}

Header::Keyword::~Keyword()
{
}

template<typename T>
void Header::Keyword::get_value(T& value) const
{
    value = boost::lexical_cast<T>(m_value);
}

template<typename T>
void Header::Keyword::set_value(const T& value)
{
    m_value = boost::lexical_cast<std::string>(value);
    m_datatype = select_fits_data_type<T>();
}

void Header::Keyword::get_comment(std::string& comment) const
{
    comment = m_comment;
}

void Header::Keyword::set_comment(const std::string& comment)
{
    m_comment = comment;
}

bool Header::has_keyword(const std::string& key)
{
    return m_keywords.find(key) != m_keywords.end();
}

void Header::del_keyword(const std::string& key)
{
    if (!has_keyword(key))
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    m_keywords.erase(key);
    m_keyword_order.erase(std::remove(m_keyword_order.begin(), m_keyword_order.end(), key), m_keyword_order.end());
}

Header::Keyword& Header::add_keyword(const std::string& key)
{
    if (has_keyword(key))
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    m_keywords.insert(std::make_pair(key, Keyword(key)));
    m_keyword_order.push_back(key);
    return m_keywords.at(key);
}

Header::Keyword& Header::get_keyword(const std::string& key)
{
    if (!has_keyword(key))
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    return m_keywords.at(key);
}

template void Header::Keyword::get_value<bool>(bool& value) const;
template void Header::Keyword::get_value<char>(char& value) const;
template void Header::Keyword::get_value<unsigned char>(unsigned char& value) const;
template void Header::Keyword::get_value<short>(short& value) const;
template void Header::Keyword::get_value<unsigned short>(unsigned short& value) const;
template void Header::Keyword::get_value<int>(int& value) const;
template void Header::Keyword::get_value<unsigned int>(unsigned int& value) const;
template void Header::Keyword::get_value<long>(long& value) const;
template void Header::Keyword::get_value<unsigned long>(unsigned long& value) const;
template void Header::Keyword::get_value<long long>(long long& value) const;
template void Header::Keyword::get_value<float>(float& value) const;
template void Header::Keyword::get_value<double>(double& value) const;
template void Header::Keyword::get_value<std::complex<float>>(std::complex<float>& value) const;
template void Header::Keyword::get_value<std::complex<double>>(std::complex<double>& value) const;
template void Header::Keyword::get_value<std::string>(std::string& value) const;

template void Header::Keyword::set_value<bool>(const bool& value);
template void Header::Keyword::set_value<char>(const char& value);
template void Header::Keyword::set_value<unsigned char>(const unsigned char& value);
template void Header::Keyword::set_value<short>(const short& value);
template void Header::Keyword::set_value<unsigned short>(const unsigned short& value);
template void Header::Keyword::set_value<int>(const int& value);
template void Header::Keyword::set_value<unsigned int>(const unsigned int& value);
template void Header::Keyword::set_value<long>(const long& value);
template void Header::Keyword::set_value<unsigned long>(const unsigned long& value);
template void Header::Keyword::set_value<long long>(const long long& value);
template void Header::Keyword::set_value<float>(const float& value);
template void Header::Keyword::set_value<double>(const double& value);
template void Header::Keyword::set_value<std::complex<float>>(const std::complex<float>& value);
template void Header::Keyword::set_value<std::complex<double>>(const std::complex<double>& value);
template void Header::Keyword::set_value<std::string>(const std::string& value);

template<typename T>
void _get_keyword(const std::string& filename, const std::string& key, T& value, char* comment)
{
    int status = 0;
    fitsfile* fp = nullptr;
    int datatype = select_fits_data_type<T>();

    fits_open_file(&fp, filename.c_str(), READONLY, &status);
    fits_read_key(fp, datatype, key.c_str(), &value, comment, &status);
    fits_close_file(fp, &status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

template<>
void _get_keyword(const std::string& filename, const std::string& key, bool& value, char* comment)
{
    int status = 0;
    fitsfile* fp = nullptr;
    int datatype = select_fits_data_type<bool>();
    int value_;

    fits_open_file(&fp, filename.c_str(), READONLY, &status);
    fits_read_key(fp, datatype, key.c_str(), &value_, comment, &status);
    fits_close_file(fp, &status);

    value = static_cast<bool>(value_);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

template<>
void _get_keyword(const std::string& filename, const std::string& key, std::string& value, char* comment)
{
    int status = 0;
    fitsfile* fp = nullptr;
    char* value_;

    fits_open_file(&fp, filename.c_str(), READONLY, &status);
    fits_read_key_longstr(fp, key.c_str(), &value_, comment, &status);
    fits_close_file(fp, &status);

    value = std::string(value_);
    fits_free_memory(value_, &status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

template<typename T>
void _set_keyword(const std::string& filename, const std::string& key, const T& value, const char* comment)
{
    int status = 0;
    fitsfile* fp = nullptr;
    int datatype = select_fits_data_type<T>();
    T value_ = value;

    fits_open_file(&fp, filename.c_str(), READWRITE, &status);
    fits_update_key(fp, datatype, key.c_str(), &value_, comment, &status);
    fits_close_file(fp, &status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

template<>
void _set_keyword(const std::string& filename, const std::string& key, const bool& value, const char* comment)
{
    int status = 0;
    fitsfile* fp = nullptr;
    int datatype = select_fits_data_type<bool>();
    int value_ = static_cast<int>(value);

    fits_open_file(&fp, filename.c_str(), READWRITE, &status);
    fits_update_key(fp, datatype, key.c_str(), &value_, comment, &status);
    fits_close_file(fp, &status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

template<>
void _set_keyword(const std::string& filename, const std::string& key, const std::string& value, const char* comment)
{
    int status = 0;
    fitsfile* fp = nullptr;

    fits_open_file(&fp, filename.c_str(), READWRITE, &status);
    fits_update_key_longstr(fp, key.c_str(), value.c_str(), comment, &status);
    fits_close_file(fp, &status);

    if (status) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}

template<typename T>
void get_keyword(const std::string& filename, const std::string& key, T& value)
{
    _get_keyword<T>(filename, key, value, nullptr);
}

template<typename T>
void get_keyword(const std::string& filename, const std::string& key, T& value, std::string& comment)
{
    char comment_[1024];
    _get_keyword<T>(filename, key, value, comment_);
    comment = std::string(comment_);
}

template<typename T>
void set_keyword(const std::string& filename, const std::string& key, const T& value)
{
    _set_keyword<T>(filename, key, value, nullptr);
}

template<typename T>
void set_keyword(const std::string& filename, const std::string& key, const T& value, const std::string& comment)
{
    _set_keyword<T>(filename, key, value, comment.c_str());
}

template void get_keyword<bool>(const std::string& filename, const std::string& key, bool& value);
template void get_keyword<char>(const std::string& filename, const std::string& key, char& value);
template void get_keyword<unsigned char>(const std::string& filename, const std::string& key, unsigned char& value);
template void get_keyword<short>(const std::string& filename, const std::string& key, short& value);
template void get_keyword<unsigned short>(const std::string& filename, const std::string& key, unsigned short& value);
template void get_keyword<int>(const std::string& filename, const std::string& key, int& value);
template void get_keyword<unsigned int>(const std::string& filename, const std::string& key, unsigned int& value);
template void get_keyword<long>(const std::string& filename, const std::string& key, long& value);
template void get_keyword<unsigned long>(const std::string& filename, const std::string& key, unsigned long& value);
template void get_keyword<long long>(const std::string& filename, const std::string& key, long long& value);
template void get_keyword<float>(const std::string& filename, const std::string& key, float& value);
template void get_keyword<double>(const std::string& filename, const std::string& key, double& value);
template void get_keyword<std::complex<float>>(const std::string& filename, const std::string& key, std::complex<float>& value);
template void get_keyword<std::complex<double>>(const std::string& filename, const std::string& key, std::complex<double>& value);
template void get_keyword<std::string>(const std::string& filename, const std::string& key, std::string& value);

template void get_keyword<bool>(const std::string& filename, const std::string& key, bool& value, std::string& comment);
template void get_keyword<char>(const std::string& filename, const std::string& key, char& value, std::string& comment);
template void get_keyword<unsigned char>(const std::string& filename, const std::string& key, unsigned char& value, std::string& comment);
template void get_keyword<short>(const std::string& filename, const std::string& key, short& value, std::string& comment);
template void get_keyword<unsigned short>(const std::string& filename, const std::string& key, unsigned short& value, std::string& comment);
template void get_keyword<int>(const std::string& filename, const std::string& key, int& value, std::string& comment);
template void get_keyword<unsigned int>(const std::string& filename, const std::string& key, unsigned int& value, std::string& comment);
template void get_keyword<long>(const std::string& filename, const std::string& key, long& value, std::string& comment);
template void get_keyword<unsigned long>(const std::string& filename, const std::string& key, unsigned long& value, std::string& comment);
template void get_keyword<long long>(const std::string& filename, const std::string& key, long long& value, std::string& comment);
template void get_keyword<float>(const std::string& filename, const std::string& key, float& value, std::string& comment);
template void get_keyword<double>(const std::string& filename, const std::string& key, double& value, std::string& comment);
template void get_keyword<std::complex<float>>(const std::string& filename, const std::string& key, std::complex<float>& value, std::string& comment);
template void get_keyword<std::complex<double>>(const std::string& filename, const std::string& key, std::complex<double>& value, std::string& comment);
template void get_keyword<std::string>(const std::string& filename, const std::string& key, std::string& value, std::string& comment);

template void set_keyword<bool>(const std::string& filename, const std::string& key, const bool& value);
template void set_keyword<char>(const std::string& filename, const std::string& key, const char& value);
template void set_keyword<unsigned char>(const std::string& filename, const std::string& key, const unsigned char& value);
template void set_keyword<short>(const std::string& filename, const std::string& key, const short& value);
template void set_keyword<unsigned short>(const std::string& filename, const std::string& key, const unsigned short& value);
template void set_keyword<int>(const std::string& filename, const std::string& key, const int& value);
template void set_keyword<unsigned int>(const std::string& filename, const std::string& key, const unsigned int& value);
template void set_keyword<long>(const std::string& filename, const std::string& key, const long& value);
template void set_keyword<unsigned long>(const std::string& filename, const std::string& key, const unsigned long& value);
template void set_keyword<long long>(const std::string& filename, const std::string& key, const long long& value);
template void set_keyword<float>(const std::string& filename, const std::string& key, const float& value);
template void set_keyword<double>(const std::string& filename, const std::string& key, const double& value);
template void set_keyword<std::complex<float>>(const std::string& filename, const std::string& key, const std::complex<float>& value);
template void set_keyword<std::complex<double>>(const std::string& filename, const std::string& key, const std::complex<double>& value);
template void set_keyword<std::string>(const std::string& filename, const std::string& key, const std::string& value);

template void set_keyword<bool>(const std::string& filename, const std::string& key, const bool& value, const std::string& comment);
template void set_keyword<char>(const std::string& filename, const std::string& key, const char& value, const std::string& comment);
template void set_keyword<unsigned char>(const std::string& filename, const std::string& key, const unsigned char& value, const std::string& comment);
template void set_keyword<short>(const std::string& filename, const std::string& key, const short& value, const std::string& comment);
template void set_keyword<unsigned short>(const std::string& filename, const std::string& key, const unsigned short& value, const std::string& comment);
template void set_keyword<int>(const std::string& filename, const std::string& key, const int& value, const std::string& comment);
template void set_keyword<unsigned int>(const std::string& filename, const std::string& key, const unsigned int& value, const std::string& comment);
template void set_keyword<long>(const std::string& filename, const std::string& key, const long& value, const std::string& comment);
template void set_keyword<unsigned long>(const std::string& filename, const std::string& key, const unsigned long& value, const std::string& comment);
template void set_keyword<long long>(const std::string& filename, const std::string& key, const long long& value, const std::string& comment);
template void set_keyword<float>(const std::string& filename, const std::string& key, const float& value, const std::string& comment);
template void set_keyword<double>(const std::string& filename, const std::string& key, const double& value, const std::string& comment);
template void set_keyword<std::complex<float>>(const std::string& filename, const std::string& key, const std::complex<float>& value, const std::string& comment);
template void set_keyword<std::complex<double>>(const std::string& filename, const std::string& key, const std::complex<double>& value, const std::string& comment);
template void set_keyword<std::string>(const std::string& filename, const std::string& key, const std::string& value, const std::string& comment);

} // namespace fits
} // namespace gbkfit
