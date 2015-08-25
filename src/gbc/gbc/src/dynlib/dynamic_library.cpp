
#include "gbc/dynlib/dynamic_library.hpp"

namespace gbc {
namespace dynlib {


dynamic_library::dynamic_library(const std::string& name)
    : m_name(name)
{
}

dynamic_library::~dynamic_library()
{
}

void dynamic_library::open(void)
{
}


} // namespace dynlib
} // namespace gbc
