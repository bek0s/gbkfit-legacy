#pragma once
#ifndef GBKFIT_PARAMETERS_HPP
#define GBKFIT_PARAMETERS_HPP

#include "gbkfit/prerequisites.hpp"
#include <boost/lexical_cast.hpp>

namespace gbkfit {

//!
//! \brief The Parameters class
//!
class Parameters
{

private:

    //!
    //! \brief The Parameter class
    //!
    class Parameter
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
        Parameter& add(const std::string& key, const T& value)
        {
            if (m_options.find(key) != m_options.end())
                throw std::runtime_error(BOOST_CURRENT_FUNCTION);
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
        Parameter& set(const std::string& key, const T& value)
        {
            if(m_options.find(key) == m_options.end())
                throw std::runtime_error(BOOST_CURRENT_FUNCTION);
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
            if (m_options.find(key) == m_options.end())
                throw std::runtime_error(BOOST_CURRENT_FUNCTION);
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
            if (m_options.find(key) == m_options.end())
                return default_value;
            return boost::lexical_cast<T>(m_options[key]);
        }

    }; // class Parameter

private:

    std::map<std::string, Parameter> m_parameters;

public:

    //!
    //! \brief add_parameter
    //! \param name
    //! \return
    //!
    Parameter& add_parameter(const std::string& name);

    //!
    //! \brief get_parameter
    //! \param name
    //! \return
    //!
    Parameter& get_parameter(const std::string& name);

}; // class Parameters

} // namespace gbkfit

#endif // GBKFIT_PARAMETERS_HPP
