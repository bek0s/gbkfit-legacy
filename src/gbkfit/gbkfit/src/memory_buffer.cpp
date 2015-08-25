
#include "gbkfit/memory_buffer.hpp"

namespace gbkfit {


memory_buffer::memory_buffer(size_type size)
    : m_size(size)
{
}

memory_buffer::~memory_buffer()
{
}

memory_buffer::size_type memory_buffer::get_size(void) const
{
    return m_size;
}

void memory_buffer::read_data(void* dst) const
{
    read_data(dst,0,get_size());
}

void memory_buffer::write_data(const void* src)
{
    write_data(src,0,get_size());
}

void memory_buffer::copy_data(const memory_buffer* src)
{
    copy_data(src,0,0,get_size());
}


} // namespace gbkfit
