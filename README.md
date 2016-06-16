
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

## Installation

TODO

## User Guide

To execute GBKFIT use the following command:

```bash
gbkfit_app_cli --config="your_configuration_file.json"
```

### GBKFIT input configuration

GBKFIT input configuration is in JSON format. The JSON format was chosen
because it is very simple and supported by a wide range of software tools.

#### The Datasets (`datasets`)

The data of the fitting procedure are defined under the `datasets` key in the
form of an array. Each element of the array has to include the following keys:
- `type`: The type of the data. This can be:
  - `flxmap`: for flux maps
  - `velmap`: for velocity maps
  - `sigmap`: for velocity dispersion maps
  - `flxcube`: for spectral cubes
- `data`: The path to the data measurements.
- `error`: The path to the 1-sigma uncertainties in the data measurements.
- `mask`: The path to a mask image.

Example:
```json
{
  "datasets": [
    {
      "type": "velmap",
      "data": "data/velmap_d.fits",
      "error": "data/velmap_e.fits",
      "mask": "data/velmap_m.fits"
    },
    {
      "type": "sigmap",
      "data": "data/sigmap_d.fits",
      "error": "data/sigmap_e.fits",
      "mask": "data/sigmap_m.fits"
    }
  ]
}
```

#### Instrument

TODO

Example:
```json
{
  "instrument": {
    "sampling": {
      "x": 1.0,
      "y": 1.0,
      "z": 30.0
    },
    "psf": {
      "type": "gaussian",
      "fwhm_x": 2.5,
      "fwhm_y": 2.5,
      "pa": 0
    },
    "lsf": {
      "type": "gaussian",
      "fwhm": 30
    }
  }
}
```

#### The Data Model (`dmodel`)

TODO

Example:
```json
{
  "dmodel": {
    "type": "gbkfit_dmodel_mmaps_cuda"
  }
}
```

#### The Galaxy Model (`gmodel`)

TODO

Example:
```json
{
  "gmodel": {
    "type": "gbkfit_gmodel_mmodel01_cuda",
    "flx_profile": "exponential",
    "vel_profile": "arctan"
  }
}
```

#### The Fitter (`fitter`)

The fitter of the fitting procedure is defined under the `fitter` key in the
form of a map. The elements of this map depend on the selected fitter which is
defined by the `type` key:
- `gbkfit.fitter.mpfit`: Uses the `gbkfit_fitter_mpfit` fitter module which
enables the following options: `ftol`, `xtol`, `gtol`, `epsfcn`, `stepfactor`,
`covtol`, `maxiter`, `maxfev`, `nprint`, `douserscale`, `nofinitecheck`.
- `gbkfit.fitter.multinest`: Uses the `gbkfit_fitter_multinest` fitter module
which enables the following options: `is`, `mmodal`, `ceff`, `nlive`, `efr`,
`tol`, `ztol`, `logzero`, `maxiter`.

For more details as to what each option does see MPFIT's and MultiNest's
documentation. For the parameters that are not present in the configuration,
GBKFIT will use default values. If any of the supplied options do not belong
to the selected fitter, they will be ignored.

Example:
```json
{
  "fitter": {
    "type": "gbkfit.fitter.mpfit",
    "maxiter": 2000
  }
}
```

#### Parameter Priors (`parameters`)

TODO

Example:
```json
{
  "parameters": [
    {"name":   "i0", "fixed": 1, "value":   1},
    {"name":   "r0", "fixed": 1, "value":  10},
    {"name":   "xo", "fixed": 0, "value":  25, "min":  0.0, "max":  50},
    {"name":   "yo", "fixed": 0, "value":  25, "min":  0.0, "max":  50},
    {"name":   "pa", "fixed": 0, "value":  45, "min":  0.0, "max":  90},
    {"name": "incl", "fixed": 0, "value":  45, "min":  5.0, "max":  85},
    {"name":   "rt", "fixed": 0, "value":   1, "min":  0.1, "max":  10},
    {"name":   "vt", "fixed": 0, "value": 150, "min":  100, "max": 350},
    {"name": "vsys", "fixed": 0, "value":   0, "min": -100, "max": 100},
    {"name": "vsig", "fixed": 0, "value":  50, "min":    5, "max": 120}
  ]
}
```
