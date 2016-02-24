#pragma once
#ifndef GBKFIT_SPREAD_FUNCTIONS_HPP
#define GBKFIT_SPREAD_FUNCTIONS_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"

namespace gbkfit {

//!
//! \brief The LineSpreadFunction class
//!
class LineSpreadFunction
{
public:
    LineSpreadFunction(void);
    virtual ~LineSpreadFunction();
    NDArrayHost* as_array(float step) const;
    NDArrayHost* as_array(int size, float step) const;
    virtual NDShape get_recommended_size(float step) const = 0;
    virtual void as_array(int size, float step, float* data) const = 0;
}; // class LineSpreadFunction

//!
//! \brief The LineSpreadFunctionArray class
//!
class LineSpreadFunctionArray : public LineSpreadFunction
{
private:
    float* m_data;
    int m_size;
    float m_step;
public:
    LineSpreadFunctionArray(float* data, int size, float step);
    ~LineSpreadFunctionArray();
    NDShape get_recommended_size(float step) const final;
    void as_array(int size, float step, float* data) const final;
}; // class LineSpreadFunctionArray

//!
//! \brief The LineSpreadFunctionGaussian class
//!
class LineSpreadFunctionGaussian : public LineSpreadFunction
{
private:
    float m_fwhm;
public:
    LineSpreadFunctionGaussian(float fwhm);
    ~LineSpreadFunctionGaussian();
    NDShape get_recommended_size(float step) const final;
    void as_array(int size, float step, float* data) const final;
}; // class LineSpreadFunctionGaussian

//!
//! \brief The LineSpreadFunctionLorentzian class
//!
class LineSpreadFunctionLorentzian : public LineSpreadFunction
{
private:
    float m_fwhm;
public:
    LineSpreadFunctionLorentzian(float fwhm);
    ~LineSpreadFunctionLorentzian();
    NDShape get_recommended_size(float step) const final;
    void as_array(int size, float step, float* data) const final;
}; // class LineSpreadFunctionLorentzian

//!
//! \brief The LineSpreadFunctionMoffat class
//!
class LineSpreadFunctionMoffat : public LineSpreadFunction
{
private:
    float m_fwhm;
    float m_beta;
public:
    LineSpreadFunctionMoffat(float fwhm, float beta = 4.765f);
    ~LineSpreadFunctionMoffat();
    NDShape get_recommended_size(float step) const final;
    void as_array(int size, float step, float* data) const final;
}; // class LineSpreadFunctionMoffat

//!
//! \brief The PointSpreadFunction class
//!
class PointSpreadFunction
{
public:
    PointSpreadFunction(void);
    virtual ~PointSpreadFunction();
    NDArrayHost* as_image(float step_x, float step_y) const;
    NDArrayHost* as_image(int size_x, int size_y, float step_x, float step_y) const;
    virtual NDShape get_recommended_size(float step_x, float step_y) const = 0;
    virtual void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const = 0;
}; // class PointSpreadFunction

//!
//! \brief The PointSpreadFunctionImage class
//!
class PointSpreadFunctionImage : public PointSpreadFunction
{
private:
    float* m_data;
    int m_size_x;
    int m_size_y;
    float m_step_x;
    float m_step_y;
public:
    PointSpreadFunctionImage(float* data, int size_x, int size_y, float step_x, float step_y);
    ~PointSpreadFunctionImage();
    NDShape get_recommended_size(float step_x, float step_y) const final;
    void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const final;
}; // class PointSpreadFunctionImage

//!
//! \brief The PointSpreadFunctionGaussian class
//!
class PointSpreadFunctionGaussian : public PointSpreadFunction
{
private:
    float m_fwhm_x;
    float m_fwhm_y;
    float m_pa;
public:
    PointSpreadFunctionGaussian(float fwhm);
    PointSpreadFunctionGaussian(float fwhm_x, float fwhm_y, float pa);
    ~PointSpreadFunctionGaussian();
    NDShape get_recommended_size(float step_x, float step_y) const final;
    void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const final;
}; // class PointSpreadFunctionGaussian

//!
//! \brief The PointSpreadFunctionLorentzian class
//!
class PointSpreadFunctionLorentzian : public PointSpreadFunction
{
private:
    float m_fwhm_x;
    float m_fwhm_y;
    float m_pa;
public:
    PointSpreadFunctionLorentzian(float fwhm);
    PointSpreadFunctionLorentzian(float fwhm_x, float fwhm_y, float pa);
    ~PointSpreadFunctionLorentzian();
    NDShape get_recommended_size(float step_x, float step_y) const final;
    void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const final;
}; // class PointSpreadFunctionLorentzian

//!
//! \brief The PointSpreadFunctionMoffat class
//!
class PointSpreadFunctionMoffat : public PointSpreadFunction
{
private:
    float m_fwhm_x;
    float m_fwhm_y;
    float m_beta;
    float m_pa;
public:
    PointSpreadFunctionMoffat(float fwhm, float beta = 4.765f);
    PointSpreadFunctionMoffat(float fwhm_x, float fwhm_y, float pa, float beta = 4.765f);
    ~PointSpreadFunctionMoffat();
    NDShape get_recommended_size(float step_x, float step_y) const final;
    void as_image(int size_x, int size_y, float step_x, float step_y, float* data) const final;
}; // class PointSpreadFunctionMoffat

} // namespace gbkfit

#endif // GBKFIT_SPREAD_FUNCTIONS_HPP
