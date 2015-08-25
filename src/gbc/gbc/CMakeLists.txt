
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbc-project)

# ==============================================================================

#find_package(foo REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/gbc/dynlib/dynamic_library.hpp

    include/gbc/noncopyable.hpp
    include/gbc/prerequisites.hpp
)

set(SOURCE_FILES
    src/dynlib/dynamic_library.cpp
)

# ==============================================================================

set(GBC_LIBRARY_STATIC $<TARGET_FILE:target_gbc_static> CACHE INTERNAL "" FORCE)
set(GBC_LIBRARY_SHARED $<TARGET_FILE:target_gbc_shared> CACHE INTERNAL "" FORCE)
set(GBC_INCLUDE_DIR $<TARGET_PROPERTY:target_gbc_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)
set(GBC_INCLUDE_DIRS ${GBC_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbc_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbc_static STATIC $<TARGET_OBJECTS:target_gbc_object>)
add_library(target_gbc_shared SHARED $<TARGET_OBJECTS:target_gbc_object>)

target_include_directories(target_gbc_object PUBLIC "include")

target_compile_definitions(target_gbc_object PRIVATE "GBC_SELF_BUILD")

target_compile_options(target_gbc_object PRIVATE "-Wall")
target_compile_options(target_gbc_object PRIVATE "-Wextra")
target_compile_options(target_gbc_object PRIVATE "-Wpedantic")
target_compile_options(target_gbc_object PRIVATE "-Wsign-conversion")

set_target_properties(target_gbc_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbc_object PROPERTIES CXX_STANDARD 11)
set_target_properties(target_gbc_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbc_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbc_static PROPERTIES OUTPUT_NAME "gbc_static")
set_target_properties(target_gbc_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbc_shared PROPERTIES OUTPUT_NAME "gbc_shared")
set_target_properties(target_gbc_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS
        target_gbc_static
        target_gbc_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
