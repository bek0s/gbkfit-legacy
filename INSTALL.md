# GBKFIT Installation Guide

This document provides instructions on how to build and install GBKFIT and its
dependencies.

GBKFIT is a modern software and it tries to make use of latest software
technologies and coding standards. As a result, it expects that its
dependencies and the software used for its compilation and installation
procedure are relatively up-to-date.

## Compiler

GBKFIT requires a C++ compiler with C++14 support.

GBKFIT has been tested with the following compilers:
- GCC 5.4 (Debian Linux, stretch)
- GCC 5.1 (CentOS 6.7)

## Build system

GBKFIT uses the CMake build system. If the version of CMake used for the build
is too old, GBKFIT will inform the user with an error message. Furthermore,
for convenience, it is recommended to install at least one of its GUI
front-ends. To install CMake follow the steps bellow:

- Linux
  - apt-get: `apt-get install cmake cmake-ncurses-gui cmake-qt-gui`
- Mac OS
  - Homebrew: `brew install cmake`
  - MacPorts: `port install cmake +gui`
    - If CMake is already installed without a GUI front-end you might need to
    run `port uninstall cmake` first
- Windows
  - Download the latest version from [here](https://www.cmake.org/) and
  install it

## Dependencies

In this section, all the details regarding GBKFIT's internal and external
extensions are discussed. You can skip most of the reading and go directly
[here](#installing-external-dependencies). However, if you encounter problems
during the compilation and installation of GBKFIT and its dependencies, make
sure you come back and read this entire section.

### Internal dependencies

GBKFIT applications and libraries have internal dependencies:

- `gbkfit`
  - none
- `gbkfit_cuda`
  - required: `gbkfit`
- `gbkfit_fftw3`
  - required: `gbkfit`
- `gbkfit_dmodel_scube`
  - required: `gbkfit`
- `gbkfit_dmodel_scube_cuda`
  - required: `gbkfit_dmodel_scube`, `gbkfit_cuda`
- `gbkfit_dmodel_scube_omp`
  - required: `gbkfit_dmodel_scube`, `gbkfit_fftw3`
- `gbkfit_dmodel_mmaps`
  - required: `gbkfit`
- `gbkfit_dmodel_mmaps_cuda`
  - required: `gbkfit_dmodel_mmaps`, `gbkfit_dmodel_scube_cuda`
- `gbkfit_dmodel_mmaps_omp`
  - required: `gbkfit_dmodel_mmaps`, `gbkfit_dmodel_scube_omp`
- `gbkfit_gmodel_gmodel1`
  - required: `gbkfit`
- `gbkfit_gmodel_gmodel1_omp`
  - required: `gbkfit_gmodel_gmodel1`
- `gbkfit_gmodel_gmodel1_cuda`
  - required: `gbkfit_gmodel_gmodel1`, `gbkfit_cuda`
- `gbkfit_fitter_mpfit`
  - required: `gbkfit`
- `gbkfit_fitter_multinest`
  - required: `gbkfit`
- `gbkfit_app_cli`
  - required: `gbkfit`
  - optional: `gbkfit_cuda`, `gbkfit_fftw3`, `gbkfit_dmodel_scube`,
  `gbkfit_dmodel_scube_cuda`, `gbkfit_dmodel_scube_omp`,
  `gbkfit_dmodel_mmaps`, `gbkfit_dmodel_mmaps_cuda`,
  `gbkfit_dmodel_mmaps_omp`, `gbkfit_gmodel_gmodel1`,
  `gbkfit_gmodel_gmodel1_omp`, `gbkfit_gmodel_gmodel1_cuda`,
  `gbkfit_fitter_mpfit`, `gbkfit_fitter_multinest`

For example: If we want to build the `gbkfit_dmodel_scube_cuda` library, we
will have to build the libraries `gbkfit_dmodel_scube` and  `gbkfit_cuda`.
We will also need to build the libraries these two depend on. Thus, we will
need to build the `gbkfit` library. If this sounds complicated, do not worry.
CMake will automatically calculate the required dependencies for each library.

### External dependencies

GBKFIT applications and libraries have external dependencies:

- `gbkfit`
  - required: [Boost C++ libraries](http://www.boost.org/),
  [CFITSIO](http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html)
- `gbkfit_cuda`
  - required: [CUDA](https://developer.nvidia.com/cuda-toolkit)
- `gbkfit_fftw3`
  - required: [FFTW3](http://www.fftw.org/)
- `gbkfit_dmodel_mmaps`
  - required: None
- `gbkfit_dmodel_mmaps_cuda`
  - required: [CUDA](https://developer.nvidia.com/cuda-toolkit)
- `gbkfit_dmodel_mmaps_omp`
  - required: [OpenMP](http://openmp.org/)
- `gbkfit_dmodel_scube`
  - required: None
- `gbkfit_dmodel_scube_cuda`
  - required: [CUDA](https://developer.nvidia.com/cuda-toolkit)
- `gbkfit_dmodel_scube_omp`
  - required: [OpenMP](http://openmp.org/), [FFTW3](http://www.fftw.org/)
- `gbkfit_gmodel_gmodel1`
  - required: None
- `gbkfit_gmodel_gmodel1_omp`
  - required: [OpenMP](http://openmp.org/)
- `gbkfit_gmodel_gmodel1_cuda`
  - required: [CUDA](https://developer.nvidia.com/cuda-toolkit)
- `gbkfit_fitter_mpfit`
  - required: None
- `gbkfit_fitter_multinest`
  - required: [MultiNest](https://ccpforge.cse.rl.ac.uk/gf/project/multinest/)

#### Installing external dependencies

Bellow are the installation instructions for all the external dependencies on
different operating systems:

- Boost C++ libraries
  - Linux
    - apt-get:
    `apt-get install libboost-program-options-dev libboost-system-dev`
  - Mac OS
    - Homebrew: `brew install boost`
    - MacPorts: `port install boost`
  - Windows
    - Download the latest version from [here](http://www.boost.org/) and
    install it
- CFITSIO
  - Linux
    - apt-get: `apt-get install libcfitsio-dev`
  - Mac OS
    - Homebrew: `brew install cfitsio`
    - MacPorts: `port install cfitsio`
  - Windows
    - Download the latest version from
    [here](http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html) and install it
- FFTW3
  - Linux
    - apt-get: `apt-get install libfftw3-dev`
  - Mac OS
    - Homebrew: `brew install fftw --with-openmp`
    - MacPorts: `port install fftw-3-single`
  - Windows
    - Download the latest version from [here](http://www.fftw.org/) and
    install it
- CUDA
  - Linux
    - apt-get: `apt-get install nvidia-cuda-toolkit`
  - Mac OS
    - Download the latest version from
    [here](https://developer.nvidia.com/cuda-toolkit) and install it
  - Windows
    - Download the latest version from
    [here](https://developer.nvidia.com/cuda-toolkit) and install it
- MultiNest
  - Download the source from
  [here](https://ccpforge.cse.rl.ac.uk/gf/project/multinest/) and install it

## Building GBKFIT

### Configuring environment variables

To help CMake find the dependencies that are not located in any of the
standard search paths of the operating system, the following environment
variables need to be defined:
- `CFITSIO_ROOT`: The location of the root directory of the installed CFITSIO
library.
- `FFTW3_ROOT`: The location of the root directory of the installed FFTW3
library.
- `MULTINEST_ROOT`. The location of the root directory of the installed
MultiNest library.

It might be useful to mention that the _root directory_ is the directory which
usually contains the `bin`, `include`, and `lib` subdirectories.

Bellow is an example of how to set an environment variable on different
operating systems:
- *nix
  - bash: `export MULTINEST_ROOT=/home/george/usr/local/multinest`
- Windows
  - Windows7+: `setx MULTINEST_ROOT "c:\libraries\multinest"`

### Running CMake

To download, build, and install GBKFIT run the following:

1. `git clone https://github.com/bek0s/gbkfit.git`
2. `cd gbkfit`
3. `mkdir build`
4. `cd build`
5. `cmake ../ -DCMAKE_INSTALL_PREFIX=~/usr/local/gbkfit`
    - CMake will now try to configure your project and locate all the
      dependencies.
6. `make`
7. `make install`

Congratulations! You just compiled and install GBKFIT successfully!

### Common build configurations

Several custom CMake options can be used to configure the build:

- `GBKFIT_BUILD_MODELS_CPU`
  - controls whether to build the CPU-based models or not
  - default value: `TRUE`
- `GBKFIT_BUILD_MODELS_GPU`
  - controls whether to build the GPU-based models or not
  - default value: `TRUE`
- `GBKFIT_BUILD_FITTER_MPFIT`
  - controls whether to build the MPFIT-based Fitter or not
  - default value: `TRUE`
- `GBKFIT_BUILD_FITTER_MULTINEST`
  - controls whether to build the MultiNest-based Fitter or not
  - default value: `TRUE`
- `GBKFIT_BUILD_APP_CLI`
  - controls whether to build the actual GBKFIT application or not
  - default value: `TRUE`

#### Example

To compile GBKFIT without GPU support, use the following CMake command:
```bash
cmake ../ -DCMAKE_INSTALL_PREFIX=~/usr/local/gbkfit -DGBKFIT_BUILD_MODELS_GPU=FALSE
```