#pragma once
#ifndef GBKFIT_SPREAD_FUNCTIONS_HPP
#define GBKFIT_SPREAD_FUNCTIONS_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"

namespace gbkfit {

//!
//! \brief The line_spread_function class
//!
class line_spread_function
{
public:
    line_spread_function(void);
    virtual ~line_spread_function();
    NDArrayHost* as_array(float step) const;
    NDArrayHost* as_array(int size, float step) const;
    virtual NDShape get_recommended_size(float step) const = 0;
    virtual void as_array(int size, float step, float* data) const = 0;
}; // class line_spread_function

//!
//! \brief The line_spread_function_array class
//!
class line_spread_function_array : public line_spread_function
{
private:
    float* m_data;
    float m_step;
    int m_size;
public:
    line_spread_function_array(float* data, float step, int size);
    ~line_spread_function_array();
    NDShape get_recommended_size(float step) const final;
    void as_array(int size, float step, float* data) const final;
}; // class line_spread_function_array

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
    NDShape get_recommended_size(float step) const final;
    void as_array(int size, float step, float* data) const final;
}; // class line_spread_function_gaussian

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
    NDShape get_recommended_size(float step) const final;
    void as_array(int size, float step, float* data) const final;
}; // class line_spread_function_lorentzian

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
    NDShape get_recommended_size(float step) const final;
    void as_array(int size, float step, float* data) const final;
}; // class line_spread_function_moffat

//!
//! \brief The point_spread_function class
//!
class point_spread_function
{
public:
    point_spread_function(void);
    virtual ~point_spread_function();
    NDArrayHost* as_image(float step_x, float step_y) const;
    NDArrayHost* as_image(int size_x, int size_y, float step_x, float step_y) const;
    virtual NDShape get_recommended_size(float step_x, float step_y) const = 0;
    virtual void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const = 0;
}; // class point_spread_function

//!
//! \brief The point_spread_function_image class
//!
class point_spread_function_image : public point_spread_function
{
private:
    float* m_data;
    float m_step_x;
    float m_step_y;
    int m_size_x;
    int m_size_y;
public:
    point_spread_function_image(float* data, int size_x, int size_y, float step_x, float step_y);
    ~point_spread_function_image();
    NDShape get_recommended_size(float step_x, float step_y) const final;
    void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const final;
}; // class point_spread_function_image

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
    NDShape get_recommended_size(float step_x, float step_y) const final;
    void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const final;
}; // point_spread_function_gaussian

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
    NDShape get_recommended_size(float step_x, float step_y) const final;
    void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const final;
}; // point_spread_function_lorentzian

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
    NDShape get_recommended_size(float step_x, float step_y) const final;
    void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const final;
}; // point_spread_function_moffat

} // namespace gbkfit

#endif // GBKFIT_SPREAD_FUNCTIONS_HPP
