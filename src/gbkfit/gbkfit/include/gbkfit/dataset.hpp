#pragma once
#ifndef GBKFIT_DATASET_HPP
#define GBKFIT_DATASET_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"

namespace gbkfit
{

//!
//! \brief The Dataset class
//!
class Dataset
{

private:

    std::string m_name;

    std::map<std::string, Data*> m_data_map;

    NDArray* m_data;
    NDArray* m_errors;
    NDArray* m_mask;


public:

    Dataset(const std::string& name);

    virtual ~Dataset();

    const std::string& get_name(void) const;

    Dataset(const std::string& name, NDArray* data, NDArray* errors, NDArray* mask);

    const NDArray* get_data(void) const;

    const NDArray* get_errors(void) const;

    const NDArray* get_mask(void) const;

}; // class Dataset

} // namespace gbkfit

#endif // GBKFIT_DATASET_HPP
