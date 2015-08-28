
#include "gbkfit/memory_buffer_host.hpp"

namespace gbkfit {


memory_buffer_host::memory_buffer_host(size_type size)
    : memory_buffer(size),
      m_data(nullptr)
{
}

memory_buffer_host::~memory_buffer_host()
{
}

void* memory_buffer_host::get_host_ptr(void)
{
    return m_data;
}

const void* memory_buffer_host::get_host_ptr(void) const
{
    return m_data;
}

void memory_buffer_host::read_data(void* dst, size_type src_offset, size_type length) const
{
    const std::uint8_t* src = m_data+src_offset;
    std::memcpy(dst,src,length);
}

void memory_buffer_host::write_data(const void* src, size_type dst_offset, size_type length)
{
    std::uint8_t* dst = m_data+dst_offset;
    std::memcpy(dst,src,length);
}

void memory_buffer_host::copy_data(const memory_buffer_host* src, size_type src_offset, size_type dst_offset, size_type length)
{
    std::uint8_t* dst = m_data+dst_offset;
    src->read_data(dst,src_offset,length);
}

void memory_buffer_host::copy_data(const memory_buffer* src, size_type src_offset, size_type dst_offset, size_type length)
{
    if(const memory_buffer_host* src_buffer = reinterpret_cast<const memory_buffer_host*>(src))
    {
        copy_data(src_buffer,src_offset,dst_offset,length);
    }
    else
    {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }
}


} // namespace gbkfit
