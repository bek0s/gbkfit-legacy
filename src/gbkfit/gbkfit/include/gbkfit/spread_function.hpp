#pragma once
#ifndef GBKFIT_SPREAD_FUNCTION_HPP
#define GBKFIT_SPREAD_FUNCTION_HPP

#include <vector>

namespace gbkfit {

//!
//! \brief The line_spread_function class
//!
class line_spread_function
{
public:
    line_spread_function(void);
    virtual ~line_spread_function();
    virtual void as_array(std::size_t length, float step, float* data) const = 0;
};

//!
//! \brief The line_spread_function_array class
//!
class line_spread_function_array : public line_spread_function
{
private:
    std::vector<float> m_data;
    std::size_t m_length;
    float m_step;
public:
    line_spread_function_array(float* data, std::size_t length, float step);
    ~line_spread_function_array();
    void as_array(std::size_t length, float step, float* data) const final;
};

//!
//! \brief The line_spread_function_gaussian class
//!
class line_spread_function_gaussian : public line_spread_function
{
private:
    float m_fwhm;
public:
    line_spread_function_gaussian(float fwhm);
    ~line_spread_function_gaussian();
    void as_array(std::size_t length, float step, float* data) const final;
};

//!
//! \brief The line_spread_function_lorentzian class
//!
class line_spread_function_lorentzian : public line_spread_function
{
private:
    float m_fwhm;
public:
    line_spread_function_lorentzian(float fwhm);
    ~line_spread_function_lorentzian();
    void as_array(std::size_t length, float step, float* data) const final;
};

//!
//! \brief The line_spread_function_moffat class
//!
class line_spread_function_moffat : public line_spread_function
{
private:
    float m_fwhm;
    float m_beta;
public:
    line_spread_function_moffat(float fwhm, float beta = 4.765f);
    ~line_spread_function_moffat();
    void as_array(std::size_t length, float step, float* data) const final;
};

//!
//! \brief The point_spread_function class
//!
class point_spread_function
{
public:
    point_spread_function(void);
    virtual ~point_spread_function();
    virtual void as_image(std::size_t length_x, std::size_t length_y, float step_x, float step_y, float* data) const = 0;
};

//!
//! \brief The point_spread_function_image class
//!
class point_spread_function_image : public point_spread_function
{
private:
    std::vector<float> m_data;
    std::size_t m_length_x;
    std::size_t m_length_y;
    float m_step_x;
    float m_step_y;
public:
    point_spread_function_image(float* data, std::size_t length_x, std::size_t length_y, float step_x, float step_y);
    ~point_spread_function_image();
    void as_image(std::size_t length_x, std::size_t length_y, float step_x, float step_y, float* data) const final;
};

//!
//! \brief The point_spread_function_gaussian class
//!
class point_spread_function_gaussian : public point_spread_function
{
private:
    float m_fwhm_x;
    float m_fwhm_y;
    float m_pa;
public:
    point_spread_function_gaussian(float fwhm);
    point_spread_function_gaussian(float fwhm_x, float fwhm_y, float pa);
    ~point_spread_function_gaussian();
    void as_image(std::size_t length_x, std::size_t length_y, float step_x, float step_y, float* data) const final;
};

//!
//! \brief The point_spread_function_lorentzian class
//!
class point_spread_function_lorentzian : public point_spread_function
{
private:
    float m_fwhm_x;
    float m_fwhm_y;
    float m_pa;
public:
    point_spread_function_lorentzian(float fwhm);
    point_spread_function_lorentzian(float fwhm_x, float fwhm_y, float pa);
    ~point_spread_function_lorentzian();
    void as_image(std::size_t length_x, std::size_t length_y, float step_x, float step_y, float* data) const final;
};

//!
//! \brief The point_spread_function_moffat class
//!
class point_spread_function_moffat : public point_spread_function
{
private:
    float m_fwhm_x;
    float m_fwhm_y;
    float m_beta;
    float m_pa;
public:
    point_spread_function_moffat(float fwhm, float beta = 4.765f);
    point_spread_function_moffat(float fwhm_x, float fwhm_y, float pa, float beta = 4.765f);
    ~point_spread_function_moffat();
    void as_image(std::size_t length_x, std::size_t length_y, float step_x, float step_y, float* data) const final;
};

} // namespace gbkfit

#endif // GBKFIT_SPREAD_FUNCTION_HPP
