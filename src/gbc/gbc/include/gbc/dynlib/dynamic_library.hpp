#pragma once
#ifndef GBC_DYNLIB_DYNAMIC_LIBRARY_HPP
#define GBC_DYNLIB_DYNAMIC_LIBRARY_HPP

#include "gbc/prerequisites.hpp"
#include "gbc/noncopyable.hpp"

namespace gbc {
namespace dynlib {


class dynamic_library : public gbc::noncopyable
{
private:
    std::string m_name;
public:
    dynamic_library(const std::string& name);
    ~dynamic_library();
    void open(void);
    void close(void);
}; // class dynamic_library


} // namespace dynlib
} // namespace gbc

#endif // GBC_DYNLIB_DYNAMIC_LIBRARY_HPP
