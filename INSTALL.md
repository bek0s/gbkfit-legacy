# GBKFIT installation instructions

This document provides instructions on how to build and install GBKFIT and its
dependencies.

GBKFIT is a modern software and it tries to make use of latest software
technologies and coding standards. As a result, it expects that its
dependencies and the software used for its compilation and installation
procedure is relatively up-to-date.

## Compiler

GBKFIT requires a C++ compiler with C++14 support.

GBKFIT has been tested with the following compilers:
- GCC 5.4 (Debian Linux, stretch repo)
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

GBKFIT is a set of libraries. Each library has its own dependencies:

- `gbkfit`
  - [Boost C++ libraries](http://www.boost.org/)
  - [CFITSIO](http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html)
- `gbkfit_cuda`
  - [CUDA](https://developer.nvidia.com/cuda-toolkit)
- `gbkfit_fftw3`
  - [FFTW3](http://www.fftw.org/)
- `gbkfit_dmodel_mmaps_omp`
  - [OpenMP](http://openmp.org/)
- `gbkfit_dmodel_mmaps_cuda`
  - [CUDA](https://developer.nvidia.com/cuda-toolkit)
- `gbkfit_dmodel_scube_omp`
  - [OpenMP](http://openmp.org/)
  - [FFTW3](http://www.fftw.org/)
- `gbkfit_dmodel_scube_cuda`
  - [CUDA](https://developer.nvidia.com/cuda-toolkit)
- `gbkfit_gmodel_omp`
  - [OpenMP](http://openmp.org/)
- `gbkfit_gmodel_cuda`
  - [CUDA](https://developer.nvidia.com/cuda-toolkit)
- `gbkfit_fitter_mpfit`
  - [MPFIT](https://www.physics.wisc.edu/~craigm/idl/cmpfit.html)
- `gbkfit_fitter_multinest`
  - [MultiNest](https://ccpforge.cse.rl.ac.uk/gf/project/multinest/)

Bellow are the installation instructions for all the dependencies on different
operating systems:

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
- MPFIT
  - Download the source from
  [here](https://www.physics.wisc.edu/~craigm/idl/cmpfit.html)
  - Linux and Mac OS
    - Edit Makefile and add the flag `-fPIC` at the C compiler flags, so the
    line will become: `$(CC) -c -o $@ $< $(CFLAGS) -fPIC`
    - Run make and then copy the library and header files to a location of
    your choice
  - Windows
    - Never tried it. If you do please let me know! :)
- MultiNest
  - Download the source from
  [here](https://ccpforge.cse.rl.ac.uk/gf/project/multinest/)
  - Compile and install it using the instructions provided

## Building GBKFIT

TODO
