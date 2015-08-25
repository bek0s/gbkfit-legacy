
#include "gbkfit/memory_buffer_malloc.hpp"

namespace gbkfit {


memory_buffer_malloc::memory_buffer_malloc(size_type size)
    : memory_buffer_host(size)
{
    m_data = reinterpret_cast<std::uint8_t*>(std::malloc(size));
}

memory_buffer_malloc::~memory_buffer_malloc(void)
{
    std::free(m_data);
}


} // namespace gbkfit
