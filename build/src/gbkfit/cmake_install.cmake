# Install script for directory: /home/bekos/code/gbkfit/gbkfit/src/gbkfit

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/cmake_install.cmake")
  include("/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit_cuda/cmake_install.cmake")
  include("/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit_fftw3/cmake_install.cmake")
  include("/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit_fitters_mpfit/cmake_install.cmake")
  include("/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit_fitters_multinest/cmake_install.cmake")
  include("/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit_models_galaxy_2d/cmake_install.cmake")
  include("/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit_models_galaxy_2d_cuda/cmake_install.cmake")
  include("/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit_models_galaxy_2d_omp/cmake_install.cmake")

endif()

