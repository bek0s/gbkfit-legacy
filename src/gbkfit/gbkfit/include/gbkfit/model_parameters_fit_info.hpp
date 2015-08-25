#pragma once
#ifndef GBKFIT_MODEL_PARAMETERS_FIT_INFO_HPP
#define GBKFIT_MODEL_PARAMETERS_FIT_INFO_HPP

#include "gbkfit/prerequisites.hpp"
#include <boost/lexical_cast.hpp>

namespace gbkfit {

//!
//! \brief The model_parameter_fit_info class
//!
class model_parameter_fit_info
{

private:

    std::map<std::string,std::string> m_options;

public:

    //!
    //! \brief has
    //! \param key
    //! \return
    //!
    bool has(const std::string& key) const
    {
        return m_options.find(key) != m_options.end();
    }

    //!
    //! \brief add
    //! \param key
    //! \param value
    //! \return
    //!
    template<typename T>
    model_parameter_fit_info& add(const std::string& key, const T& value)
    {
        if(m_options.find(key) != m_options.end())
            throw std::runtime_error("key already exists");
        m_options[key] = boost::lexical_cast<std::string>(value);
        return *this;
    }

    //!
    //! \brief set
    //! \param key
    //! \param value
    //! \return
    //!
    template<typename T>
    model_parameter_fit_info& set(const std::string& key, const T& value)
    {
        if(m_options.find(key) == m_options.end())
            throw std::runtime_error("key does not exist");
        m_options[key] = boost::lexical_cast<std::string>(value);
        return *this;
    }

    //!
    //! \brief get
    //! \param key
    //! \return
    //!
    template<typename T>
    T get(const std::string& key)
    {
        if(m_options.find(key) == m_options.end())
            throw std::runtime_error("key does not exist");
        return boost::lexical_cast<T>(m_options[key]);
    }

    //!
    //! \brief get
    //! \param key
    //! \param default_value
    //! \return
    //!
    template<typename T>
    T get(const std::string& key, const T& default_value)
    {
        if(m_options.find(key) == m_options.end())
            return default_value;
        return boost::lexical_cast<T>(m_options[key]);
    }

}; // class model_parameter_fit_info


//!
//! \brief The model_parameters_fit_info class
//!
class model_parameters_fit_info
{

private:

    std::map<std::string,model_parameter_fit_info> m_parameters;

public:

    //!
    //! \brief add_parameter
    //! \param name
    //! \return
    //!
    model_parameter_fit_info& add_parameter(const std::string& name);

    //!
    //! \brief get_parameter
    //! \param name
    //! \return
    //!
    model_parameter_fit_info& get_parameter(const std::string& name);

}; // class model_parameters_fit_info


} // namespace gbkfit

#endif // GBKFIT_MODEL_PARAMETERS_FIT_INFO_HPP
