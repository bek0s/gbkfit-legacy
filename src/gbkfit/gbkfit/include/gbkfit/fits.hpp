#pragma once
#ifndef GBKFIT_FITS_UTIL_HPP
#define GBKFIT_FITS_UTIL_HPP

#include "gbkfit/prerequisites.hpp"
#include <boost/lexical_cast.hpp>

namespace gbkfit {
namespace fits {

//!
//! \brief The header class
//!
class header
{

public:

    std::vector<std::string> m_key_names;
    std::vector<std::string> m_key_values;
    std::vector<std::string> m_key_comments;

    std::map<std::string,std::size_t> m_key_indices;

public:

    bool has_key(const std::string& name) const
    {
        return m_key_indices.find(name) != m_key_indices.end();
    }

    template<typename T>
    void set_key(const std::string& name, const T& value, const std::string& comment)
    {
        std::string value_str = boost::lexical_cast<std::string>(value);
        m_key_names.push_back(name);
        m_key_values.push_back(value_str);
    }

    template<typename T>
    void get_key(const std::string& name, T& value, std::string& comment)
    {}

    template<typename T>
    T get_key_value(const std::string& name) const
    {}

    template<typename T>
    void set_key_value(const std::string& name, const T& value)
    {}

    std::string get_key_comment(const std::string& keyname, std::string& comment) const
    {}

    void set_key_comment(const std::string& keyname, const std::string& comment)
    {}

}; // class header


//!
//! \brief get_header
//! \param filename
//! \return
//!
header get_header(const std::string& filename);

//!
//! \brief get_data
//! \param filename
//! \return
//!
ndarray* get_data(const std::string& filename);

//!
//! \brief write_to
//! \param filename
//! \param data
//!
void write_to(const std::string& filename, const ndarray& data);

//!
//! \brief get_key
//! \param filename
//! \param keyname
//! \param value
//! \param comment
//!
template<typename T>
void get_key(const std::string& filename, const std::string& keyname, T& value, std::string& comment);

//!
//! \brief get_key_value
//! \param filename
//! \param keyname
//! \param value
//!
template<typename T>
void get_key_value(const std::string& filename, const std::string& keyname, T& value);

//!
//! \brief get_key_comment
//! \param filename
//! \param keyname
//! \param comment
//!
void get_key_comment(const std::string& filename, const std::string& keyname, std::string& comment);

//!
//! \brief set_key
//! \param filename
//! \param keyname
//! \param value
//! \param comment
//!
template<typename T>
void set_key(const std::string& filename, const std::string& keyname, const T& value, const std::string& comment);

//!
//! \brief set_key_value
//! \param filename
//! \param keyname
//! \param keyvalue
//!
template<typename T>
void set_key_value(const std::string& filename, const std::string& keyname, const T& value);

//!
//! \brief set_key_comment
//! \param filename
//! \param keyname
//! \param comment
//!
void set_key_comment(const std::string& filename, const std::string& keyname, const std::string& comment);

//!
//! \brief del_key
//! \param filename
//! \param keyname
//!
void del_key(const std::string& filename, const std::string& keyname);

} // namespace fits
} // namespace gbkfit

#endif // GBKFIT_FITS_UTIL_HPP
