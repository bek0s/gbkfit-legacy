#pragma once
#ifndef GBKFIT_FITS_UTIL_HPP
#define GBKFIT_FITS_UTIL_HPP

#include "gbkfit/prerequisites.hpp"
#include <boost/lexical_cast.hpp>

namespace gbkfit {

namespace fits {


enum class value_type
{
    tnull,
    tbit,
    tubyte,
    tbyte,
    tbool,
    tstring,
    tushort,
    tshort,
    tuint,
    tint,
    tulong,
    tlong,
    tlonglong,
    tfloat,
    tdouble,
    tcomplexfloat,
    tcomplexdouble,



};

enum class image_type
{
    tnull,
    tbyte,
    tshort,
    tlong,
    tlonglong,
    tfloat,
    tdouble
};

class keyword_base
{

public:

    std::string m_name;
    std::string m_comment;
    value_type m_value_type;

protected:


    keyword_base(const std::string& name, value_type type)
    {
        m_name = name;
        m_value_type = type;
    }

};

template <typename T> value_type select_value_type();





class attribute
{

public:

    enum class data_type
    {
        adt_null,
        adt_bool,
        adt_int8,
        adt_uint8,
        adt_int16,
        adt_uint16,
        adt_int32,
        adt_uint32,
        adt_int64,
        adt_uint64,
        adt_float32,
        adt_float64,
        adt_complex32,
        adt_complex64,
        adt_string
    };

public:

    std::string m_name;
    std::string m_comment;
    std::string m_value;
    data_type m_data_type;

public:

    //attribute()

    template<typename T>
    T get_value(void) const;

    template<typename T>
    void set_value(const T& value);

private:


};

/*

template <typename T> attribute::data_type select_data_type();
template <> attribute::data_type select_data_type<bool>(void) { return attribute::data_type::adt_bool; }
template <> attribute::data_type select_data_type<std::int8_t>(void) { return attribute::data_type::adt_int8; }
template <> attribute::data_type select_data_type<std::uint8_t>(void) { return attribute::data_type::adt_uint8; }
template <> attribute::data_type select_data_type<std::int16_t>(void) { return attribute::data_type::adt_int16; }
template <> attribute::data_type select_data_type<std::uint16_t>(void) { return attribute::data_type::adt_uint16; }
template <> attribute::data_type select_data_type<std::int32_t>(void) { return attribute::data_type::adt_int32; }
template <> attribute::data_type select_data_type<std::uint32_t>(void) { return attribute::data_type::adt_uint32; }
template <> attribute::data_type select_data_type<std::int64_t>(void) { return attribute::data_type::adt_int64; }
template <> attribute::data_type select_data_type<std::uint64_t>(void) { return attribute::data_type::adt_uint64; }
template <> attribute::data_type select_data_type<float>(void) { return attribute::data_type::adt_float32; }
template <> attribute::data_type select_data_type<double>(void) { return attribute::data_type::adt_float64; }
template <> attribute::data_type select_data_type<std::complex<float>>(void) { return attribute::data_type::adt_complex32; }
template <> attribute::data_type select_data_type<std::complex<double>>(void) { return attribute::data_type::adt_complex64; }
template <> attribute::data_type select_data_type<std::string>(void) { return attribute::data_type::adt_string; }
*/


template<typename T>
T attribute::get_value(void) const
{
    return boost::lexical_cast<T>(m_value);
}

template<typename T>
void attribute::set_value(const T& value)
{
    m_value = boost::lexical_cast<std::string>(value);
    //m_data_type = select_data_type<T>();
}






template<typename T>
class keyword : public keyword_base
{
public:

    keyword(const std::string& name, T value)
        : keyword_base(name,select_value_type<T>())
    {

    }
};

class header
{

public:

    enum class key_data_type
    {
        float32,
        float64
    };



public:



    std::vector<std::string> m_key_names;
    std::vector<std::string> m_key_values;
    std::vector<std::string> m_key_comments;
    std::vector<key_data_type> m_key_data_type;

    std::map<std::string,std::size_t> m_key_indices;



public:

    bool has_key(const std::string& name) const
    {
        return m_key_indices.find(name) != m_key_indices.end();
    }

    template<typename T>
    void set_key(const std::string& name, const T& value, const std::string& comment = "")
    {
        std::string value_str = boost::lexical_cast<std::string>(value);

        m_key_names.push_back(name);
        m_key_values.push_back(value_str);
    }


    template<typename T>
    void get_key(const std::string& name, T& value, std::string& comment)
    {
    }


    template<typename T>
    T get_key_value(const std::string& name) const
    {
        //auto m_key_indices.find(name);
    }

    template<typename T>
    void set_key_value(const std::string& name, const T& value)
    {
    }

};


header get_header(const std::string& name);




//!
//! \brief get_data
//! \param filename
//! \return
//!
ndarray* get_data2(const std::string& filename);



//!
//! \brief get_keyword
//! \param filename
//! \param keyname
//! \param value
//! \param comment
//!
template<typename T>
void get_keyword(const std::string& filename, const std::string& keyname, T& value, std::string& comment);

//!
//! \brief get_keyword_value
//! \param filename
//! \param keyname
//! \param value
//!
template<typename T>
void get_keyword_value(const std::string& filename, const std::string& keyname, T& value);

//!
//! \brief get_keyword_comment
//! \param filename
//! \param keyname
//! \param comment
//!
void get_keyword_comment(const std::string& filename, const std::string& keyname, std::string& comment);

//!
//! \brief set_keyword
//! \param filename
//! \param keyname
//! \param value
//! \param comment
//!
template<typename T>
void set_keyword(const std::string& filename, const std::string& keyname, const T& value, const std::string& comment);

//!
//! \brief set_keyword_value
//! \param filename
//! \param keyname
//! \param keyvalue
//!
template<typename T>
void set_keyword_value(const std::string& filename, const std::string& keyname, const T& value);

//!
//! \brief set_keyword_comment
//! \param filename
//! \param keyname
//! \param comment
//!
void set_keyword_comment(const std::string& filename, const std::string& keyname, const std::string& comment);

//!
//! \brief del_keyword
//! \param filename
//! \param keyname
//!
void del_keyword(const std::string& filename, const std::string& keyname);


}


namespace fits_util {


void get_header(const std::string& name);

ndarray* get_data(const std::string& name);

void write_to(const std::string& name, const ndarray& data);

void write_to2(const std::string& name, const ndarray& data);





} // namespace fits_utils
} // namespace gbkfit

#endif // GBKFIT_FITS_UTIL_HPP