
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_app_cli-project)

# ==============================================================================

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

add_executable(gbkfit_app_cli
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

# ==============================================================================

if(NOT ${GBKFIT_BUILD_CUDA})
    unset(GBKFIT_CUDA_INCLUDE_DIR CACHE)
    unset(GBKFIT_CUDA_LIBRARY_SHARED CACHE)
endif()

if(NOT ${GBKFIT_BUILD_FITTER_MULTINEST})
    unset(GBKFIT_FITTER_MULTINEST_INCLUDE_DIR CACHE)
    unset(GBKFIT_FITTER_MULTINEST_LIBRARY_SHARED CACHE)
endif()

if(NOT ${GBKFIT_BUILD_FITTER_MPFIT})
    unset(GBKFIT_FITTER_MPFIT_INCLUDE_DIR CACHE)
    unset(GBKFIT_FITTER_MPFIT_LIBRARY_SHARED CACHE)
endif()

if(NOT ${GBKFIT_BUILD_DMODEL})
    unset(GBKFIT_DMODEL_INCLUDE_DIR CACHE)
    unset(GBKFIT_DMODEL_LIBRARY_SHARED CACHE)
endif()

if(NOT ${GBKFIT_BUILD_DMODEL_OMP})
    unset(GBKFIT_DMODEL_OMP_INCLUDE_DIR CACHE)
    unset(GBKFIT_DMODEL_OMP_LIBRARY_SHARED CACHE)
endif()

if(NOT ${GBKFIT_BUILD_DMODEL_CUDA})
    unset(GBKFIT_DMODEL_CUDA_INCLUDE_DIR CACHE)
    unset(GBKFIT_DMODEL_CUDA_LIBRARY_SHARED CACHE)
endif()

if(NOT ${GBKFIT_BUILD_GMODEL_GMODEL1})
    unset(GBKFIT_GMODEL_GMODEL1_INCLUDE_DIR CACHE)
    unset(GBKFIT_GMODEL_GMODEL1_LIBRARY_SHARED CACHE)
endif()

if(NOT ${GBKFIT_BUILD_GMODEL_GMODEL1_OMP})
    unset(GBKFIT_GMODEL_GMODEL1_OMP_INCLUDE_DIR CACHE)
    unset(GBKFIT_GMODEL_GMODEL1_OMP_LIBRARY_SHARED CACHE)
endif()

if(NOT ${GBKFIT_BUILD_GMODEL_GMODEL1_CUDA})
    unset(GBKFIT_GMODEL_GMODEL1_CUDA_INCLUDE_DIR CACHE)
    unset(GBKFIT_GMODEL_GMODEL1_CUDA_LIBRARY_SHARED CACHE)
endif()

# ==============================================================================

target_include_directories(gbkfit_app_cli
    PRIVATE
    "include"
    ${GBKFIT_INCLUDE_DIR}
    ${GBKFIT_CUDA_INCLUDE_DIR}
    ${GBKFIT_FITTER_MPFIT_INCLUDE_DIR}
    ${GBKFIT_FITTER_MULTINEST_INCLUDE_DIR}
    ${GBKFIT_DMODEL_INCLUDE_DIR}
    ${GBKFIT_DMODEL_OMP_INCLUDE_DIR}
    ${GBKFIT_DMODEL_CUDA_INCLUDE_DIR}
    ${GBKFIT_GMODEL_INCLUDE_DIR}
    ${GBKFIT_GMODEL_OMP_INCLUDE_DIR}
    ${GBKFIT_GMODEL_CUDA_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(gbkfit_app_cli
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
    ${GBKFIT_CUDA_LIBRARY_SHARED}
    ${GBKFIT_FITTER_MPFIT_LIBRARY_SHARED}
    ${GBKFIT_FITTER_MULTINEST_LIBRARY_SHARED}
    ${GBKFIT_DMODEL_LIBRARY_SHARED}
    ${GBKFIT_DMODEL_OMP_LIBRARY_SHARED}
    ${GBKFIT_DMODEL_CUDA_LIBRARY_SHARED}
    ${GBKFIT_GMODEL_LIBRARY_SHARED}
    ${GBKFIT_GMODEL_OMP_LIBRARY_SHARED}
    ${GBKFIT_GMODEL_CUDA_LIBRARY_SHARED}
    ${Boost_LIBRARIES}
)

# ==============================================================================

target_compile_options(gbkfit_app_cli
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(gbkfit_app_cli
    PROPERTIES
    CXX_EXTENSIONS        OFF
    CXX_STANDARD          14
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME           "gbkfit_app_cli"
    DEBUG_POSTFIX         "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_app_cli
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# ==============================================================================
