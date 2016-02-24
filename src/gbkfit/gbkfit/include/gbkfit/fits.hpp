#pragma once
#ifndef GBKFIT_FITS_UTIL_HPP
#define GBKFIT_FITS_UTIL_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {
namespace fits {

//!
//! \brief The Header class
//!
class Header
{

public:

    //!
    //! \brief The Keyword class
    //!
    class Keyword
    {

    private:

        std::string m_key;
        std::string m_value;
        std::string m_comment;
        int m_datatype;

    public:

        Keyword(const std::string& key);

        ~Keyword();

        template<typename T>
        void get(T& value, std::string& comment)
        {
            get_value<T>(value);
            get_comment(comment);
        }

        template<typename T>
        void set(const T& value, const std::string& comment)
        {
            set_value<T>(value);
            set_comment(comment);
        }

        template<typename T>
        void get_value(T& value) const;

        template<typename T>
        void set_value(const T& value);

        void get_comment(std::string& comment) const;

        void set_comment(const std::string& comment);

    }; // class Keyword

public:

    std::vector<std::string> m_keyword_order;
    std::map<std::string, Keyword> m_keywords;

public:

    bool has_keyword(const std::string& key);

    void del_keyword(const std::string& key);

    Keyword& add_keyword(const std::string& key);

    Keyword& get_keyword(const std::string& key);

}; // class Header

//!
//! \brief get_header
//! \param filename
//! \return
//!
Header get_header(const std::string& filename);

std::unique_ptr<NDArrayHost> get_data(const std::string& filename);

void write_to(const std::string& filename, const NDArray& data);

//!
//! \brief get_keyword
//! \param filename
//! \param key
//! \param value
//!
template<typename T>
void get_keyword(const std::string& filename, const std::string& key, T& value);

//!
//! \brief get_keyword
//! \param filename
//! \param key
//! \param value
//! \param comment
//!
template<typename T>
void get_keyword(const std::string& filename, const std::string& key, T& value, std::string& comment);

//!
//! \brief set_keyword
//! \param filename
//! \param key
//! \param value
//!
template<typename T>
void set_keyword(const std::string& filename, const std::string& key, const T& value);

//!
//! \brief set_keyword
//! \param filename
//! \param key
//! \param value
//! \param comment
//!
template<typename T>
void set_keyword(const std::string& filename, const std::string& key, const T& value, const std::string& comment);

} // namespace fits
} // namespace gbkfit

#endif // GBKFIT_FITS_UTIL_HPP
