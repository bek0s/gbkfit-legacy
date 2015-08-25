#pragma once
#ifndef GBKFIT_NDDATASET_HPP
#define GBKFIT_NDDATASET_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit
{


//!
//! \brief The nddataset class
//!
//! This is a placeholder class. It will change it in the near future.
//! Yes, the syntax is ugly, I know.
//!
class nddataset
{

private:

    std::map<std::string,ndarray*> m_data_map;

public:

    nddataset(void);

    virtual ~nddataset();

    std::map<std::string,ndarray*>& get(void);

}; // class nddataset


} // namespace gbkfit

#endif // GBKFIT_NDDATASET_HPP
