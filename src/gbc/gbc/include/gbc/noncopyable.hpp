#pragma once
#ifndef GBC_CORE_NONCOPYABLE_HPP
#define GBC_CORE_NONCOPYABLE_HPP

namespace gbc
{

class noncopyable
{
protected:
    constexpr noncopyable() = default;
    ~noncopyable() = default;
    noncopyable(const noncopyable&) = delete;
    noncopyable& operator=(const noncopyable&) = delete;
}; // class noncopyable

} // namespace gbc

#endif // GBC_CORE_NONCOPYABLE_HPP
