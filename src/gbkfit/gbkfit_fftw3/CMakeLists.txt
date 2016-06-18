
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_fftw3-project)

# ==============================================================================

find_package(fftw3)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/fftw3/ndarray.hpp
    include/gbkfit/fftw3/util.hpp
)

set(SOURCE_FILES
    src/ndarray.cpp
    src/util.cpp
)

# ==============================================================================

set(GBKFIT_FFTW3_LIBRARY_STATIC $<TARGET_FILE:gbkfit_fftw3_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_FFTW3_LIBRARY_SHARED $<TARGET_FILE:gbkfit_fftw3_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_FFTW3_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_fftw3_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_fftw3_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_fftw3_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_fftw3_object>
)

add_library(gbkfit_fftw3_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_fftw3_object>
)

# ==============================================================================

target_include_directories(gbkfit_fftw3_object
    PUBLIC
    "include"
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIRS}
)

target_link_libraries(gbkfit_fftw3_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
    ${FFTW3_SINGLE_THREADS_LIBRARY}
)

target_link_libraries(gbkfit_fftw3_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
    ${FFTW3_SINGLE_THREADS_LIBRARY}
)

target_compile_definitions(gbkfit_fftw3_object
    PRIVATE
    "GBKFIT_FFTW3_SELF_BUILD"
)

target_compile_options(gbkfit_fftw3_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(gbkfit_fftw3_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_fftw3_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_fftw3_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_fftw3_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_fftw3_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_fftw3_static
    gbkfit_fftw3_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
