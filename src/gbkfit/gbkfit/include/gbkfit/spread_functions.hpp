#pragma once
#ifndef GBKFIT_SPREAD_FUNCTIONS_HPP
#define GBKFIT_SPREAD_FUNCTIONS_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/ndshape.hpp"

namespace gbkfit {

class LineSpreadFunction
{
public:
    LineSpreadFunction(void);
    LineSpreadFunction(const LineSpreadFunction&) = delete;
    LineSpreadFunction& operator=(const LineSpreadFunction&) = delete;
    virtual ~LineSpreadFunction();
    std::unique_ptr<NDArrayHost> as_array(float step) const;
    std::unique_ptr<NDArrayHost> as_array(float step, int size) const;
    virtual NDShape get_size(float step) const = 0;
    virtual void as_array(float step, int size, float* data) const = 0;
};

class LineSpreadFunctionNone : public LineSpreadFunction
{
public:
    LineSpreadFunctionNone(void);
    ~LineSpreadFunctionNone();
    NDShape get_size(float step) const override final;
    void as_array(float step, int size, float* data) const override final;
};

class LineSpreadFunctionArray : public LineSpreadFunction
{
private:
    float* m_data;
    int m_size;
public:
    LineSpreadFunctionArray(const float* data, int size);
    ~LineSpreadFunctionArray();
    NDShape get_size(float step) const override final;
    void as_array(float step, int size, float* data) const override final;
};

class LineSpreadFunctionGaussian : public LineSpreadFunction
{
private:
    float m_fwhm;
public:
    LineSpreadFunctionGaussian(float fwhm);
    ~LineSpreadFunctionGaussian();
    NDShape get_size(float step) const override final;
    void as_array(float step, int size, float* data) const override final;
};

class LineSpreadFunctionLorentzian : public LineSpreadFunction
{
private:
    float m_fwhm;
public:
    LineSpreadFunctionLorentzian(float fwhm);
    ~LineSpreadFunctionLorentzian();
    NDShape get_size(float step) const override final;
    void as_array(float step, int size, float* data) const override final;
};

class LineSpreadFunctionMoffat : public LineSpreadFunction
{
private:
    float m_fwhm;
    float m_beta;
public:
    LineSpreadFunctionMoffat(float fwhm, float beta = 4.765f);
    ~LineSpreadFunctionMoffat();
    NDShape get_size(float step) const override final;
    void as_array(float step, int size, float* data) const override final;
};

class PointSpreadFunction
{
public:
    PointSpreadFunction(void);
    PointSpreadFunction(const PointSpreadFunction&) = delete;
    PointSpreadFunction& operator=(const PointSpreadFunction&) = delete;
    virtual ~PointSpreadFunction();
    std::unique_ptr<NDArrayHost> as_image(float step_x, float step_y) const;
    std::unique_ptr<NDArrayHost> as_image(float step_x, float step_y, int size_x, int size_y) const;
    virtual NDShape get_size(float step_x, float step_y) const = 0;
    virtual void as_image(float step_x, float step_y, int size_x, int size_y, float* data) const = 0;
};

class PointSpreadFunctionNone : public PointSpreadFunction
{
public:
    PointSpreadFunctionNone(void);
    ~PointSpreadFunctionNone();
    NDShape get_size(float step_x, float step_y) const override final;
    void as_image(float step_x, float step_y, int size_x, int size_y, float* data) const override final;
};

class PointSpreadFunctionImage : public PointSpreadFunction
{
private:
    float* m_data;
    int m_size_x;
    int m_size_y;
public:
    PointSpreadFunctionImage(const float* data, int size_x, int size_y);
    ~PointSpreadFunctionImage();
    NDShape get_size(float step_x, float step_y) const override final;
    void as_image(float step_x, float step_y, int size_x, int size_y, float* data) const override final;
};

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
    NDShape get_size(float step_x, float step_y) const override final;
    void as_image(float step_x, float step_y, int size_x, int size_y, float* data) const override final;
};

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
    NDShape get_size(float step_x, float step_y) const override final;
    void as_image(float step_x, float step_y, int size_x, int size_y, float* data) const override final;
};

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
    NDShape get_size(float step_x, float step_y) const override final;
    void as_image(float step_x, float step_y, int size_x, int size_y, float* data) const override final;
};

namespace spread_function_util
{

NDShape get_psf_cube_size(const PointSpreadFunction *psf,
                          const LineSpreadFunction *lsf,
                          float step_x,
                          float step_y,
                          float step_z);

std::unique_ptr<NDArrayHost> create_psf_cube(
        const PointSpreadFunction* psf,
        const LineSpreadFunction* lsf,
        float step_x,
        float step_y,
        float step_z);

std::unique_ptr<NDArrayHost> create_psf_cube(
        const PointSpreadFunction* psf,
        const LineSpreadFunction* lsf,
        float step_x,
        float step_y,
        float step_z,
        int size_x,
        int size_y,
        int size_z);

} // namespace spread_function_util

} // namespace gbkfit

#endif // GBKFIT_SPREAD_FUNCTIONS_HPP
