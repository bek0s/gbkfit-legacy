# GBKFIT User Guide

To execute GBKFIT use the following command:

```bash
gbkfit_app_cli --config="your_configuration_file.json"
```

## GBKFIT input configuration

GBKFIT input configuration is in JSON format. The JSON format was chosen
because it is very simple, flexible, and supported by a wide range of software
tools. If you are performing an automated or batch analysis with GBKFIT, it is
advised to use a JSON library to generate the configuration file. For example,
if your scripts are in Python, use the `json` module that comes with Python by
default.

### Running mode (`mode`)

The running mode of GBKFIT is specified by the `mode` key and it can have the
following values:
- `fit`: Perform a fit
- `evaluate`: Evaluate a model and store it to the disk without performing a
fit.

Example:
```json
{
  "mode": "fit"
}
```

### The Datasets (`datasets`)

The data of the fitting procedure are specified under the `datasets` array.
Each element of the array is a map and can include the following keys:
- `type`: The type of the data. This can be:
  - `flxmap`: for flux maps
  - `velmap`: for velocity maps
  - `sigmap`: for velocity dispersion maps
  - `flxcube`: for spectral cubes
- `data`: The path to the data measurements.
- `error`: The path to the 1-sigma uncertainties in the data measurements.
- `mask`: The path to a mask image.

In case of FITS files, the file names can utilize the [Extended File Name
Syntax](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/filters.html) of
the CFITSIO library.

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
      "data": "data/sigmap.fits[0]",
      "error": "data/sigmap.fits[1]",
      "mask": "data/sigmap.fits[2]"
    }
  ]
}
```

### The Point Spread Function (`psf`)

The PSF of the observation is specified under the `psf` map. Under this map,
the key `type` defines the model used for the PSF. Each model comes with its
own set of parameters:

- `gaussian`
  - `fhwm_x`
  - `fhwm_y`
  - `pa`
- `lorentzian`
  - `fhwm_x`
  - `fhwm_y`
  - `pa`
- `moffat`
  - `fhwm_x`
  - `fhwm_y`
  - `pa`
  - `beta`
- `image`

Example:
```json
{
  "psf": {
    "type": "gaussian",
    "fwhm_x": 2.5,
    "fwhm_y": 2.5,
    "pa": 0
  }
}
```

### The Line Spread Function (`lsf`)

The LSF of the observation is specified under the `lsf` map. Under this map,
the key `type` defines the model used for the LSF. Each model comes with its
own set of parameters:

- `gaussian`
  - `fhwm`
- `lorentzian`
  - `fhwm`
- `moffat`
  - `fhwm`
  - `beta`
- `image`

example:
```json
{
  "lsf": {
    "type": "gaussian",
    "fwhm": 30
  }
}
```

### The data model (`dmodel`)

The spatial sampling of the input data (specified under the `datasets` array)
are defined using the `step[0]` and `step[1]` keys, and they can be in
any units. However, the same units have to be used for any of the model
parameters that describe spatial position or distance. The spectral sampling
of the input data (specified under the `datasets` array) is defined using the
`step[2]` key and it should be in `km/s`.

TODO

Example:
```json
{
  "dmodel": {
    "type": "gbkfit.dmodel.mmaps_omp",
    "size": [49, 49, 101],
    "step": [1.0, 1.0, 1.0],
    "upsampling": [1, 1, 1]
  }
}
```

### The galaxy model (`gmodel`)

TODO

Example:
```json
{
  "gmodel": {
    "type": "gbkfit.gmodel.gmodel1_omp",
    "flx_profile": "exponential",
    "vel_profile": "arctan"
  }
}
```

### The Fitter (`fitter`)

The Fitter of the fitting procedure is defined under the `fitter` map. The
elements of this map depend on the selected Fitter which is specified by the
`type` key:
- `gbkfit.fitter.mpfit`: It is provided by the `gbkfit_fitter_mpfit` module
and enables the following options: `ftol`, `xtol`, `gtol`, `epsfcn`,
`stepfactor`, `covtol`, `maxiter`, `maxfev`, `nprint`, `douserscale`,
`nofinitecheck`.
- `gbkfit.fitter.multinest`: It is provided by the `gbkfit_fitter_multinest`
module and enables the following options: `is`, `mmodal`, `ceff`, `nlive`,
`efr`, `tol`, `ztol`, `logzero`, `maxiter`.

For more details as to what each option does and what is its default value (in
case it is optional and not present in the configuration), see the
corresponding optimizer's documentation. If any of the supplied options is not
supported by the selected Fitter, it will be ignored.

Example:
```json
{
  "fitter": {
    "type": "gbkfit.fitter.mpfit",
    "maxiter": 2000
  }
}
```

### Parameter priors (`params`)

The priors and settings for each of the model parameters are defined under the
`params` array. Each element of the array is a map and corresponds to a
different parameter. Each map can contain different items depending on running
mode or Fitter used:

- Mode: `evaluate`
  - `name`
    - Required: yes
    - Description: The name of the parameter.
  - `value`
    - Required: yes
    - Description: The value of the parameter.
- Fitter: `gbkfit.fitter.mpfit`
  - `name`
    - Required: yes
    - Description: The name of the parameter.
  - `value`
    - Required: yes
    - Description: The initial guess of the parameter value.
  - `fixed`
    - Required: no
    - Description: Set to `0` to keep the parameter free or `1` to fix the
    parameter to `value`.
  - `min`
    - Required: no
    - Description: The minimum possible value of the parameter. If not
  provided, `-std::numeric_limits<float>::max` is used.
  - `max`
    - Required: no
    - Description: The maximum possible value of the parameter. If not
  provided, `+std::numeric_limits<float>::max` is used.
  - `side`
    - Required: no
    - Description: See `MPFIT` documentation.
  - `step`
    - Required: no
    - Description: See `MPFIT` documentation.
  - `relstep`
    - Required: no
    - Description: See `MPFIT` documentation.

- Fitter: `gbkfit.fitter.multinest`
  - `name`
    - Required: yes
    - Description: The name of the parameter.
  - `value`
    - Required: no unless fixed.
    - Description: The value of the parameter if it is fixed (see `fixed`
      option bellow).
  - `fixed`
    - Required: no
    - Description: Set to `0` to keep the parameter free or `1` to fix the
      parameter to `value`.
  - `min`
    - Required: yes if not fixed
    - Description: The minimum possible value of the parameter. If not
    provided, the value `-std::numeric_limits<float>::max` is used.
  - `max`
    - Required: yes if not fixed
    - Description: The maximum possible value of the parameter. If not
    provided, the value `+std::numeric_limits<float>::max` is used.
  - `wrap`
    - Required: no
    - Description: See `MultiNest` documentation.

For more details as to what each option does and what is its default value (in
case it is optional and not present in the configuration), see the
corresponding optimizer's documentation. If any of the supplied options is not
supported by the selected Fitter, it will be ignored.

Example:
```json
{
  "params": [
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
