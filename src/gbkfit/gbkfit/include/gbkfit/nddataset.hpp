#pragma once
#ifndef GBKFIT_DATASET_HPP
#define GBKFIT_DATASET_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit
{

//!
//! \brief The Dataset class
//!
class Dataset
{

private:

    std::string m_type;
    std::map<std::string, NDArray*> m_data_map;

public:

    Dataset(const std::string& type);

    virtual ~Dataset();

    const std::string& get_type(void) const;

    bool has_data(const std::string& name) const;

    NDArray* get_data(const std::string& name);

    const NDArray* get_data(const std::string& name) const;

    void add_data(const std::string& name, NDArray* data);

}; // class Dataset


typedef std::map<std::string, Dataset*> Datasets;


} // namespace gbkfit

#endif // GBKFIT_DATASET_HPP
