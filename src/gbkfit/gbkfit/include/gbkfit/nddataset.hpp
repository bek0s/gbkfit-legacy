#pragma once
#ifndef GBKFIT_NDDATASET_HPP
#define GBKFIT_NDDATASET_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit
{


//!
//! \brief The nddataset class
//!
class nddataset
{

private:

    std::map<std::string,ndarray*> m_data_map;

public:

    nddataset(void);

    virtual ~nddataset();

    bool has_data(const std::string& name) const;

    ndarray* get_data(const std::string& name);

    const ndarray* get_data(const std::string& name) const;

    void add_data(const std::string& name, ndarray* data);

    void __destroy(void);

}; // class nddataset


} // namespace gbkfit

#endif // GBKFIT_NDDATASET_HPP
