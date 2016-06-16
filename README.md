
# GBKFIT

GBKFIT is a collection of libraries and executables for modelling galaxy
kinematics from 3D spectroscopic observations.

## Credits

GBKFIT is developed by Georgios Bekiaris (Swinburne University of Technology).

If you use GBKFIT in a publication please cite:
[Bekiaris et al. 2016](http://adsabs.harvard.edu/abs/2016MNRAS.455..754B).

## A brief introduction

Galaxy kinematic modelling is a computationally intensive process and it can
result very long run times. GBKFIT tackles this problem by utilizing the
many-core architectures of modern computers. By default, GBKFIT will
accelerate the likelihood evaluation step of the fitting procedure on the
Graphics Processing Unit (GPU). If there is no GPU available on the system, it
will use all the cores of the Central Processing Unit (CPU).

GBKFIT features a modular architecture which allows it to use a variety of
data models (e.g., spectral cubes for 3D fitting, moment maps for 2D fitting,
etc), galaxy models, and optimization techniques. It also provides a clean
object-oriented interface which enables programmers to create and add their
own custom models and optimization techniques into the software.

GBKFIT models observations with a combination of two models: a Data Model
(DModel), and a Galaxy Model (GModel). The former is used to describe the data
structure of the observation, while the latter is used to describe the
observed galaxy. Data and galaxy models come in the form of modules, and by
convention their names start with `gbkfit_dmodel_` and `gbkfit_gmodel_`
respectively.

Similarly to data and galaxy models, optimization techniques come in the form
of modules which are called Fitters, and by convention their names start with
`gbkfit_fitter_`.

### Data models

GBKFIT comes with the following data models:
- `gbkfit_dmodel_mmaps`: This model is used to describe moment maps extracted
from a spectral data cube. Thus, this model should be used to perform 2D fits
to velocity and velocity dispersion maps. Flux maps are also supported but
they are currently experimental and should not be used.
- `gbkfit_dmodel_scube`: This model is used to describe spectral data cubes.
Thus, this model should be used to perform 3D fits to spectral data cubes.
Support for 3D fitting is experimental and should be avoided.

### Galaxy models

GBKFIT comes with the following galaxy models:
- `gbkfit_gmodel_gmodel01`: This model is a combination of a thin and flat
disk, a surface brightness profile, a rotation curve, and an intrinsic
velocity dispersion which is assumed to be constant across the galactic disk.

  The following surface brightness profiles are supported:
  - Exponential disk

  The following rotation curve profiles are supported:
  - Linear ramp
  - Arctan
  - Boissier
  - Epinat

### Fitters

GBKFIT comes with the following fitters:
- `gbkfit_fitter_mpfit`: This fitter employs the Levenberg-Marquardt Algorithm
through the [MPFIT](https://www.physics.wisc.edu/~craigm/idl/cmpfit.html)
library.
- `gbkfit_fitter_multinest`: This fitter employs the Clustered and Importance
Nested Sampling techniques through the
[MultiNest](https://ccpforge.cse.rl.ac.uk/gf/project/multinest/) library.


### Point Spread Functions

GBKFIT supports the following Point Spread Function (PSF) models:
- 2D elliptical Gaussian
- 2D elliptical Lorentzian
- 2D elliptical Moffat
- 2D image

### Line Spread Functions

GBKFIT supports the following Line Spread Function (LSF) models:
- 1D Gaussian
- 1D Lorentzian
- 1D Moffat
- 1D image

## User Guide

Coming soon!
