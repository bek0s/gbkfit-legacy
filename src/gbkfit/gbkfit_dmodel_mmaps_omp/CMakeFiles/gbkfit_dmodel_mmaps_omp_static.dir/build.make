# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bekos/code/gbkfit/gbkfit

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bekos/code/gbkfit/gbkfit

# Include any dependencies generated for this target.
include src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/depend.make

# Include the progress variables for this target.
include src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/progress.make

# Include the compile flags for this target's objects.
include src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/flags.make

# Object files for target gbkfit_dmodel_mmaps_omp_static
gbkfit_dmodel_mmaps_omp_static_OBJECTS =

# External object files for target gbkfit_dmodel_mmaps_omp_static
gbkfit_dmodel_mmaps_omp_static_EXTERNAL_OBJECTS = \
"/home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_object.dir/src/mmaps_omp.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_object.dir/src/mmaps_omp_factory.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_object.dir/src/mmaps_omp_kernels.cpp.o"

lib/libgbkfit_dmodel_mmaps_omp_static.a: src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_object.dir/src/mmaps_omp.cpp.o
lib/libgbkfit_dmodel_mmaps_omp_static.a: src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_object.dir/src/mmaps_omp_factory.cpp.o
lib/libgbkfit_dmodel_mmaps_omp_static.a: src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_object.dir/src/mmaps_omp_kernels.cpp.o
lib/libgbkfit_dmodel_mmaps_omp_static.a: src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/build.make
lib/libgbkfit_dmodel_mmaps_omp_static.a: src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bekos/code/gbkfit/gbkfit/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library ../../../lib/libgbkfit_dmodel_mmaps_omp_static.a"
	cd /home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_dmodel_mmaps_omp && $(CMAKE_COMMAND) -P CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/cmake_clean_target.cmake
	cd /home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_dmodel_mmaps_omp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/build: lib/libgbkfit_dmodel_mmaps_omp_static.a

.PHONY : src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/build

src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/requires:

.PHONY : src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/requires

src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/clean:
	cd /home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_dmodel_mmaps_omp && $(CMAKE_COMMAND) -P CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/cmake_clean.cmake
.PHONY : src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/clean

src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/depend:
	cd /home/bekos/code/gbkfit/gbkfit && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bekos/code/gbkfit/gbkfit /home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_dmodel_mmaps_omp /home/bekos/code/gbkfit/gbkfit /home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_dmodel_mmaps_omp /home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/depend

