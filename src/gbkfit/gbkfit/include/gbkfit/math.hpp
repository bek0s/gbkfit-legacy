#pragma once
#ifndef GBKFIT_MATH_HPP
#define GBKFIT_MATH_HPP

#include <cmath>

namespace gbkfit {
namespace math {

template<typename T>
constexpr T pi(void)
{
    return std::atan(static_cast<T>(1))*4;
}

template<typename T>
T rad_to_deg(T rad)
{
    return rad * (180 / pi<T>());
}

template<typename T>
T deg_to_rad(T deg)
{
    return deg * (pi<T>() / 180);
}

template<typename T>
T gaussian_function_1d(T x, T amplitude, T mu, T sigma)
{
    return amplitude*std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
}

template<typename T>
T gaussian_function_1d_normalized(T x, T mu, T sigma)
{
    T amplitude = 1/(sigma*std::sqrt(2*math::pi<T>()));
    return gaussian_function_1d(x,amplitude,mu,sigma);
}

template<typename T>
T gaussian_function_2d(T x, T y, T amplitude, T mu_x, T mu_y, T sigma_x, T sigma_y, T phi)
{
    T cosphi = std::cos(phi);
    T sinphi = std::sin(phi);
    T sigma_x2 = std::pow(sigma_x,2);
    T sigma_y2 = std::pow(sigma_y,2);
    T a = std::pow(cosphi/sigma_x,2)+std::pow(sinphi/sigma_y,2);
    T b = std::pow(sinphi/sigma_x,2)+std::pow(cosphi/sigma_y,2);
    T c = 2*sinphi*cosphi*(1/sigma_x2-1/sigma_y2);
    return amplitude*std::exp(-0.5*(a*std::pow(x-mu_x,2)+b*std::pow(y-mu_y,2)+c*(x-mu_x)*(y-mu_y)));
}

template<typename T>
T gaussian_fwhm_from_sigma(T sigma)
{
    return sigma * static_cast<T>(2.35482);
}

template<typename T>
T gaussian_sigma_from_fwhm(T fwhm)
{
    return fwhm / static_cast<T>(2.35482);
}

template<typename T>
T moffat_function_1d(T x, T amplitude, T x0, T alpha, T beta)
{
    return amplitude*std::pow(1+((x-x0)/alpha)*((x-x0)/alpha),-beta);
}

template<typename T>
T moffat_function_2d(T x, T y, T amplitude, T x0, T y0, T alpha_x, T alpha_y, T beta, T phi)
{
    T cosphi = std::cos(phi);
    T sinphi = std::sin(phi);
    T alpha_x2 = std::pow(alpha_x,2);
    T alpha_y2 = std::pow(alpha_y,2);
    T a = std::pow(cosphi/alpha_x,2) + std::pow(sinphi/alpha_y,2);
    T b = std::pow(sinphi/alpha_x,2) + std::pow(cosphi/alpha_y,2);
    T c = 2*sinphi*cosphi*(1/alpha_x2 - 1/alpha_y2);
    return amplitude * std::pow(1 + a*std::pow(x-x0,2) + b*std::pow(y-y0,2) + c*(x-x0)*(y-y0),-beta);
}

template<typename T>
T moffat_fwhm_from_beta_and_alpha(T beta, T alpha)
{
    return 2*alpha*std::sqrt(std::pow(2,1/beta)-1);
}

template<typename T>
T moffat_alpha_from_beta_and_fwhm(T beta, T fwhm)
{
    return fwhm/(2*std::sqrt(std::pow(2,1/beta)-1));
}

template<typename T>
T lorentzian_function_1d(T x, T a, T b, T c)
{
    return a * ((c*c)/((x-b)*(x-b)+c*c));
}

template<typename T>
T lorentzian_function_1d_normalized(T x, T b, T c)
{
    T a = 1.0/(pi<T>()*c);
    return lorentzian_function_1d(x,a,b,c);
}

template<typename T>
T lorentzian_function_2d(T x, T y, T amplitude, T x0, T y0, T gamma_x, T gamma_y, T phi)
{
    T cosphi = std::cos(phi);
    T sinphi = std::sin(phi);
    T gamma_x2 = std::pow(gamma_x,2);
    T gamma_y2 = std::pow(gamma_y,2);
    T a = std::pow(cosphi/gamma_x,2)+std::pow(sinphi/gamma_y,2);
    T b = std::pow(sinphi/gamma_x,2)+std::pow(cosphi/gamma_y,2);
    T c = 2*sinphi*cosphi*(1/gamma_x2-1/gamma_y2);
    return amplitude * amplitude * 1 / (1 + a*std::pow(x-x0,2) + b*std::pow(y-y0,2) + c*(x-x0)*(y-y0));
}

template<typename T>
T lorentzian_fwhm_from_gamma(T gamma)
{
    return gamma * 2.0f;
}

template<typename T>
T lorentzian_gamma_from_fwhm(T fwhm)
{
    return fwhm / 2.0f;
}

template<typename T>
void normalize_integral(T* data, int length)
{
    T sum = std::accumulate(data,data+length,static_cast<T>(0));
    std::for_each(data,data+length,[&sum](float &val)->void{val/=sum;});
}

} // namespace math
} // namespace gbkfit

#endif // GBKFIT_MATH_HPP
