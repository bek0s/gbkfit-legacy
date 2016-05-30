
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_app_cli-project)

# ==============================================================================

find_package(cfitsio)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost COMPONENTS program_options system REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/application.hpp
    include/prerequisites.hpp
)

set(SOURCE_FILES
    src/application.cpp
    src/main.cpp
)

# ==============================================================================

add_executable(target_gbkfit_app_cli ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(target_gbkfit_app_cli
    PRIVATE
    "include"
    ${GBKFIT_INCLUDE_DIRS}
    ${GBKFIT_CUDA_INCLUDE_DIRS}
    ${GBKFIT_FITTER_MPFIT_INCLUDE_DIR}
    ${GBKFIT_FITTER_MULTINEST_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${GBKFIT_DMODEL_MMAPS_CUDA_INCLUDE_DIR}
    ${GBKFIT_DMODEL_MMAPS_OMP_INCLUDE_DIR}
    ${GBKFIT_DMODEL_SCUBE_CUDA_INCLUDE_DIR}
    ${GBKFIT_DMODEL_SCUBE_OMP_INCLUDE_DIR}
    ${GBKFIT_GMODEL_GMODEL01_INCLUDE_DIR}
    ${GBKFIT_GMODEL_GMODEL01_CUDA_INCLUDE_DIR}
    ${GBKFIT_GMODEL_GMODEL01_OMP_INCLUDE_DIR}
)

target_link_libraries(target_gbkfit_app_cli
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
    ${GBKFIT_CUDA_LIBRARY_SHARED}
    ${GBKFIT_FITTER_MPFIT_LIBRARY_SHARED}
    ${GBKFIT_FITTER_MULTINEST_LIBRARY_SHARED}
    ${Boost_LIBRARIES}
    fftw3f
    ${GBKFIT_DMODEL_MMAPS_CUDA_LIBRARY_SHARED}
    ${GBKFIT_DMODEL_MMAPS_OMP_LIBRARY_SHARED}
    ${GBKFIT_DMODEL_SCUBE_CUDA_LIBRARY_SHARED}
    ${GBKFIT_DMODEL_SCUBE_OMP_LIBRARY_SHARED}
    ${GBKFIT_GMODEL_GMODEL01_LIBRARY_SHARED}
    ${GBKFIT_GMODEL_GMODEL01_CUDA_LIBRARY_SHARED}
    ${GBKFIT_GMODEL_GMODEL01_OMP_LIBRARY_SHARED}
)

target_compile_options(target_gbkfit_app_cli
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(target_gbkfit_app_cli
    PROPERTIES
    CXX_EXTENSIONS        OFF
    CXX_STANDARD          14
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME           "gbkfit_app_cli"
    DEBUG_POSTFIX         "_d"
)

# ==============================================================================

install(
    TARGETS
    target_gbkfit_app_cli
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# ==============================================================================
