#pragma once
#ifndef GBKFIT_DMODEL_MMAPS_MMAPS_HPP
#define GBKFIT_DMODEL_MMAPS_MMAPS_HPP

#include "gbkfit/dmodel.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {

enum MomentMethod {
    moments = 1,
    gaussian = 2,
    hermite = 3
};

class MMaps : public DModel
{
};

} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_MMAPS_MMAPS_HPP
