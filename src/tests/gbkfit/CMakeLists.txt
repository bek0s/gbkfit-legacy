
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(ut_gbkfit-project)

# ==============================================================================

enable_testing()
find_package(GTest REQUIRED)

# ==============================================================================

set(HEADER_FILES
)

set(SOURCE_FILES
    gbkfit_fits.cpp
)

# ==============================================================================

add_executable(target_ut_gbkfit ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(target_ut_gbkfit PRIVATE ${GBKFIT_INCLUDE_DIRS})
target_include_directories(target_ut_gbkfit PRIVATE ${GTEST_INCLUDE_DIRS})

target_link_libraries(target_ut_gbkfit PRIVATE ${GBKFIT_LIBRARY_SHARED})
target_link_libraries(target_ut_gbkfit PRIVATE ${GTEST_BOTH_LIBRARIES} pthread)

target_compile_options(target_ut_gbkfit PRIVATE "-Wall")
target_compile_options(target_ut_gbkfit PRIVATE "-Wextra")
target_compile_options(target_ut_gbkfit PRIVATE "-Wpedantic")
target_compile_options(target_ut_gbkfit PRIVATE "-Wsign-conversion")

set_target_properties(target_ut_gbkfit PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_ut_gbkfit PROPERTIES CXX_STANDARD 14)
set_target_properties(target_ut_gbkfit PROPERTIES CXX_STANDARD_REQUIRED ON)

set_target_properties(target_ut_gbkfit PROPERTIES OUTPUT_NAME "target_ut_gbkfit")
set_target_properties(target_ut_gbkfit PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS
        target_ut_gbkfit
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

# ==============================================================================
