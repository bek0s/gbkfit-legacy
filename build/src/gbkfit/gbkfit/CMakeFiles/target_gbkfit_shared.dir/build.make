# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

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
CMAKE_BINARY_DIR = /home/bekos/code/gbkfit/gbkfit/build

# Include any dependencies generated for this target.
include src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/depend.make

# Include the progress variables for this target.
include src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/progress.make

# Include the compile flags for this target's objects.
include src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/flags.make

# Object files for target target_gbkfit_shared
target_gbkfit_shared_OBJECTS =

# External object files for target target_gbkfit_shared
target_gbkfit_shared_EXTERNAL_OBJECTS = \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/convolver.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/core.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/datamodel.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/fitter.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/fits.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/galmodel.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/gbkfit.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/image_util.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/instrument.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/model.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/ndarray.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/ndarray_host.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/nddataset.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/ndshape.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/spread_function.cpp.o" \
"/home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/parameters_fit_info.cpp.o"

lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/convolver.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/core.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/datamodel.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/fitter.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/fits.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/galmodel.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/gbkfit.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/image_util.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/instrument.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/model.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/ndarray.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/ndarray_host.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/nddataset.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/ndshape.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/spread_function.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_object.dir/src/parameters_fit_info.cpp.o
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/build.make
lib/libgbkfit_shared_d.so: /usr/lib/x86_64-linux-gnu/libcfitsio.so
lib/libgbkfit_shared_d.so: src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library ../../../lib/libgbkfit_shared_d.so"
	cd /home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/target_gbkfit_shared.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/build: lib/libgbkfit_shared_d.so
.PHONY : src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/build

src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/requires:
.PHONY : src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/requires

src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/clean:
	cd /home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit && $(CMAKE_COMMAND) -P CMakeFiles/target_gbkfit_shared.dir/cmake_clean.cmake
.PHONY : src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/clean

src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/depend:
	cd /home/bekos/code/gbkfit/gbkfit/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bekos/code/gbkfit/gbkfit /home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit /home/bekos/code/gbkfit/gbkfit/build /home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit /home/bekos/code/gbkfit/gbkfit/build/src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/gbkfit/gbkfit/CMakeFiles/target_gbkfit_shared.dir/depend

