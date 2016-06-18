
# Mock observations information

This text provides information about the simulated data that accompany gbkfit.

-------------------------------------------------------------------------------

## Models

The model used to create the three simulated observations is a combination of
a thin disk, a photometric profile, a rotation curve, and a intrinsic velocity
dispersion.

### The surface brightness profile

I(r) = i0 \* exp(-r/r0)

where i0 is the brightness at the galactic center, and r0 the scale length.

### The arctan rotation curve

V(r) = (2/pi) \* vt \* arctan(r/rt)

where rt is the turn-over radius, and vt the asymptotic maximum circular
velocity.

### The rest of the model parameters

- xo: RA of the galactic center.
- yo: DEC of the galactic center.
- pa: The position angle of the projected galactic disc.
- incl: The inclination of the galactic disc with respect to the line of sight.
- vsys: The systemic velocity of the galaxy.
- vsig: The intrinsic velocity dispersion which assumed to be constant across
        the galactic disc.

-------------------------------------------------------------------------------

## Model parameter values

All simulated observations share the same values for the following model
parameters:

- i0 = 1
- r0 = 10
- xo = 24.5
- yo = 24.5
- pa = 45
- rt = 4
- vt = 200
- vsys = 0
- vsig = 50

Each mock observation has a different inclination:

- mock_1: incl = 20 degrees
- mock_2: incl = 45 degrees
- mock_3: incl = 70 degrees

-------------------------------------------------------------------------------

## Instrument

All mock observations have the same dimensions, PSF, LSF and sampling:

- Width: 49 pixels
- Height: 49 pixels
- PSF: Gaussian, fwhm=2.5 pixels
- LSF: Gaussian, fwhm=30 km/s
- Spatial sampling: 1.0 pixels
- Spectral sampling: 10.0 km/s per pixel

-------------------------------------------------------------------------------

## Data files

For each mock observation the following are available:

- Flux cube (flxcube). Gbkfit uses spectral cubes (aka flux cubes) to perform 
3D fitting. However, 3D fitting support is experimental. Please use 2D fitting 
for now.
- Flux map (flxmap). Gbkfit can make use of flux maps for both 2D and 3D 
fitting. However, their support is experimental. Please ignore them for now.
- Velocity map (velmap).
- Velocity dispersion map (sigmap).

The files "lsf.fits" and "psf.fits" correspond to the LSF and PSF respectively, 
as described in the Instrument Section.

The following naming convention is used for all FITS files:

- Files ending with "_d.fits": measurements
- Files ending with "_e.fits": measurement errors
- Files ending with "_m.fits": mask

-------------------------------------------------------------------------------

## Noise

All mock observations include Gaussian noise:

- Velocity map noise: 2 km/s
- Velocity dispersion map noise: 3 km/s

There is a version of the data with no noise under the directory "clean". This 
data were created using gbkfit, and it is used by the script "create_data.py" 
in order to generate the noisy data.
