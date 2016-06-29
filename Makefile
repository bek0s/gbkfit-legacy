# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components

.PHONY : list_install_components/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: install/strip

.PHONY : install/strip/fast

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: install/local

.PHONY : install/local/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/bekos/code/gbkfit/gbkfit/CMakeFiles /home/bekos/code/gbkfit/gbkfit/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/bekos/code/gbkfit/gbkfit/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named uninstall

# Build rule for target.
uninstall: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 uninstall
.PHONY : uninstall

# fast build rule for target.
uninstall/fast:
	$(MAKE) -f CMakeFiles/uninstall.dir/build.make CMakeFiles/uninstall.dir/build
.PHONY : uninstall/fast

#=============================================================================
# Target rules for targets named gbkfit_object

# Build rule for target.
gbkfit_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_object
.PHONY : gbkfit_object

# fast build rule for target.
gbkfit_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit/CMakeFiles/gbkfit_object.dir/build.make src/gbkfit/gbkfit/CMakeFiles/gbkfit_object.dir/build
.PHONY : gbkfit_object/fast

#=============================================================================
# Target rules for targets named gbkfit_shared

# Build rule for target.
gbkfit_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_shared
.PHONY : gbkfit_shared

# fast build rule for target.
gbkfit_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit/CMakeFiles/gbkfit_shared.dir/build.make src/gbkfit/gbkfit/CMakeFiles/gbkfit_shared.dir/build
.PHONY : gbkfit_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_static

# Build rule for target.
gbkfit_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_static
.PHONY : gbkfit_static

# fast build rule for target.
gbkfit_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit/CMakeFiles/gbkfit_static.dir/build.make src/gbkfit/gbkfit/CMakeFiles/gbkfit_static.dir/build
.PHONY : gbkfit_static/fast

#=============================================================================
# Target rules for targets named gbkfit_cuda_object

# Build rule for target.
gbkfit_cuda_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_cuda_object
.PHONY : gbkfit_cuda_object

# fast build rule for target.
gbkfit_cuda_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_cuda/CMakeFiles/gbkfit_cuda_object.dir/build.make src/gbkfit/gbkfit_cuda/CMakeFiles/gbkfit_cuda_object.dir/build
.PHONY : gbkfit_cuda_object/fast

#=============================================================================
# Target rules for targets named gbkfit_cuda_static

# Build rule for target.
gbkfit_cuda_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_cuda_static
.PHONY : gbkfit_cuda_static

# fast build rule for target.
gbkfit_cuda_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_cuda/CMakeFiles/gbkfit_cuda_static.dir/build.make src/gbkfit/gbkfit_cuda/CMakeFiles/gbkfit_cuda_static.dir/build
.PHONY : gbkfit_cuda_static/fast

#=============================================================================
# Target rules for targets named gbkfit_cuda_shared

# Build rule for target.
gbkfit_cuda_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_cuda_shared
.PHONY : gbkfit_cuda_shared

# fast build rule for target.
gbkfit_cuda_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_cuda/CMakeFiles/gbkfit_cuda_shared.dir/build.make src/gbkfit/gbkfit_cuda/CMakeFiles/gbkfit_cuda_shared.dir/build
.PHONY : gbkfit_cuda_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_fftw3_object

# Build rule for target.
gbkfit_fftw3_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_fftw3_object
.PHONY : gbkfit_fftw3_object

# fast build rule for target.
gbkfit_fftw3_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_fftw3/CMakeFiles/gbkfit_fftw3_object.dir/build.make src/gbkfit/gbkfit_fftw3/CMakeFiles/gbkfit_fftw3_object.dir/build
.PHONY : gbkfit_fftw3_object/fast

#=============================================================================
# Target rules for targets named gbkfit_fftw3_shared

# Build rule for target.
gbkfit_fftw3_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_fftw3_shared
.PHONY : gbkfit_fftw3_shared

# fast build rule for target.
gbkfit_fftw3_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_fftw3/CMakeFiles/gbkfit_fftw3_shared.dir/build.make src/gbkfit/gbkfit_fftw3/CMakeFiles/gbkfit_fftw3_shared.dir/build
.PHONY : gbkfit_fftw3_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_fftw3_static

# Build rule for target.
gbkfit_fftw3_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_fftw3_static
.PHONY : gbkfit_fftw3_static

# fast build rule for target.
gbkfit_fftw3_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_fftw3/CMakeFiles/gbkfit_fftw3_static.dir/build.make src/gbkfit/gbkfit_fftw3/CMakeFiles/gbkfit_fftw3_static.dir/build
.PHONY : gbkfit_fftw3_static/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_static

# Build rule for target.
gbkfit_dmodel_scube_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_static
.PHONY : gbkfit_dmodel_scube_static

# fast build rule for target.
gbkfit_dmodel_scube_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube/CMakeFiles/gbkfit_dmodel_scube_static.dir/build.make src/gbkfit/gbkfit_dmodel_scube/CMakeFiles/gbkfit_dmodel_scube_static.dir/build
.PHONY : gbkfit_dmodel_scube_static/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_object

# Build rule for target.
gbkfit_dmodel_scube_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_object
.PHONY : gbkfit_dmodel_scube_object

# fast build rule for target.
gbkfit_dmodel_scube_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube/CMakeFiles/gbkfit_dmodel_scube_object.dir/build.make src/gbkfit/gbkfit_dmodel_scube/CMakeFiles/gbkfit_dmodel_scube_object.dir/build
.PHONY : gbkfit_dmodel_scube_object/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_shared

# Build rule for target.
gbkfit_dmodel_scube_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_shared
.PHONY : gbkfit_dmodel_scube_shared

# fast build rule for target.
gbkfit_dmodel_scube_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube/CMakeFiles/gbkfit_dmodel_scube_shared.dir/build.make src/gbkfit/gbkfit_dmodel_scube/CMakeFiles/gbkfit_dmodel_scube_shared.dir/build
.PHONY : gbkfit_dmodel_scube_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_cuda_object

# Build rule for target.
gbkfit_dmodel_scube_cuda_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_cuda_object
.PHONY : gbkfit_dmodel_scube_cuda_object

# fast build rule for target.
gbkfit_dmodel_scube_cuda_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_object.dir/build.make src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_object.dir/build
.PHONY : gbkfit_dmodel_scube_cuda_object/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_cuda_static

# Build rule for target.
gbkfit_dmodel_scube_cuda_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_cuda_static
.PHONY : gbkfit_dmodel_scube_cuda_static

# fast build rule for target.
gbkfit_dmodel_scube_cuda_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_static.dir/build.make src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_static.dir/build
.PHONY : gbkfit_dmodel_scube_cuda_static/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_cuda_nvcc_shared

# Build rule for target.
gbkfit_dmodel_scube_cuda_nvcc_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_cuda_nvcc_shared
.PHONY : gbkfit_dmodel_scube_cuda_nvcc_shared

# fast build rule for target.
gbkfit_dmodel_scube_cuda_nvcc_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_nvcc_shared.dir/build.make src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_nvcc_shared.dir/build
.PHONY : gbkfit_dmodel_scube_cuda_nvcc_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_cuda_shared

# Build rule for target.
gbkfit_dmodel_scube_cuda_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_cuda_shared
.PHONY : gbkfit_dmodel_scube_cuda_shared

# fast build rule for target.
gbkfit_dmodel_scube_cuda_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_shared.dir/build.make src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_shared.dir/build
.PHONY : gbkfit_dmodel_scube_cuda_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_cuda_nvcc_static

# Build rule for target.
gbkfit_dmodel_scube_cuda_nvcc_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_cuda_nvcc_static
.PHONY : gbkfit_dmodel_scube_cuda_nvcc_static

# fast build rule for target.
gbkfit_dmodel_scube_cuda_nvcc_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_nvcc_static.dir/build.make src/gbkfit/gbkfit_dmodel_scube_cuda/CMakeFiles/gbkfit_dmodel_scube_cuda_nvcc_static.dir/build
.PHONY : gbkfit_dmodel_scube_cuda_nvcc_static/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_omp_static

# Build rule for target.
gbkfit_dmodel_scube_omp_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_omp_static
.PHONY : gbkfit_dmodel_scube_omp_static

# fast build rule for target.
gbkfit_dmodel_scube_omp_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube_omp/CMakeFiles/gbkfit_dmodel_scube_omp_static.dir/build.make src/gbkfit/gbkfit_dmodel_scube_omp/CMakeFiles/gbkfit_dmodel_scube_omp_static.dir/build
.PHONY : gbkfit_dmodel_scube_omp_static/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_omp_object

# Build rule for target.
gbkfit_dmodel_scube_omp_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_omp_object
.PHONY : gbkfit_dmodel_scube_omp_object

# fast build rule for target.
gbkfit_dmodel_scube_omp_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube_omp/CMakeFiles/gbkfit_dmodel_scube_omp_object.dir/build.make src/gbkfit/gbkfit_dmodel_scube_omp/CMakeFiles/gbkfit_dmodel_scube_omp_object.dir/build
.PHONY : gbkfit_dmodel_scube_omp_object/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_scube_omp_shared

# Build rule for target.
gbkfit_dmodel_scube_omp_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_scube_omp_shared
.PHONY : gbkfit_dmodel_scube_omp_shared

# fast build rule for target.
gbkfit_dmodel_scube_omp_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_scube_omp/CMakeFiles/gbkfit_dmodel_scube_omp_shared.dir/build.make src/gbkfit/gbkfit_dmodel_scube_omp/CMakeFiles/gbkfit_dmodel_scube_omp_shared.dir/build
.PHONY : gbkfit_dmodel_scube_omp_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_object

# Build rule for target.
gbkfit_dmodel_mmaps_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_object
.PHONY : gbkfit_dmodel_mmaps_object

# fast build rule for target.
gbkfit_dmodel_mmaps_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps/CMakeFiles/gbkfit_dmodel_mmaps_object.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps/CMakeFiles/gbkfit_dmodel_mmaps_object.dir/build
.PHONY : gbkfit_dmodel_mmaps_object/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_static

# Build rule for target.
gbkfit_dmodel_mmaps_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_static
.PHONY : gbkfit_dmodel_mmaps_static

# fast build rule for target.
gbkfit_dmodel_mmaps_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps/CMakeFiles/gbkfit_dmodel_mmaps_static.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps/CMakeFiles/gbkfit_dmodel_mmaps_static.dir/build
.PHONY : gbkfit_dmodel_mmaps_static/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_shared

# Build rule for target.
gbkfit_dmodel_mmaps_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_shared
.PHONY : gbkfit_dmodel_mmaps_shared

# fast build rule for target.
gbkfit_dmodel_mmaps_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps/CMakeFiles/gbkfit_dmodel_mmaps_shared.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps/CMakeFiles/gbkfit_dmodel_mmaps_shared.dir/build
.PHONY : gbkfit_dmodel_mmaps_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_cuda_static

# Build rule for target.
gbkfit_dmodel_mmaps_cuda_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_cuda_static
.PHONY : gbkfit_dmodel_mmaps_cuda_static

# fast build rule for target.
gbkfit_dmodel_mmaps_cuda_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_static.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_static.dir/build
.PHONY : gbkfit_dmodel_mmaps_cuda_static/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_cuda_nvcc_static

# Build rule for target.
gbkfit_dmodel_mmaps_cuda_nvcc_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_cuda_nvcc_static
.PHONY : gbkfit_dmodel_mmaps_cuda_nvcc_static

# fast build rule for target.
gbkfit_dmodel_mmaps_cuda_nvcc_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_nvcc_static.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_nvcc_static.dir/build
.PHONY : gbkfit_dmodel_mmaps_cuda_nvcc_static/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_cuda_object

# Build rule for target.
gbkfit_dmodel_mmaps_cuda_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_cuda_object
.PHONY : gbkfit_dmodel_mmaps_cuda_object

# fast build rule for target.
gbkfit_dmodel_mmaps_cuda_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_object.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_object.dir/build
.PHONY : gbkfit_dmodel_mmaps_cuda_object/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_cuda_nvcc_shared

# Build rule for target.
gbkfit_dmodel_mmaps_cuda_nvcc_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_cuda_nvcc_shared
.PHONY : gbkfit_dmodel_mmaps_cuda_nvcc_shared

# fast build rule for target.
gbkfit_dmodel_mmaps_cuda_nvcc_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_nvcc_shared.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_nvcc_shared.dir/build
.PHONY : gbkfit_dmodel_mmaps_cuda_nvcc_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_cuda_shared

# Build rule for target.
gbkfit_dmodel_mmaps_cuda_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_cuda_shared
.PHONY : gbkfit_dmodel_mmaps_cuda_shared

# fast build rule for target.
gbkfit_dmodel_mmaps_cuda_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_shared.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps_cuda/CMakeFiles/gbkfit_dmodel_mmaps_cuda_shared.dir/build
.PHONY : gbkfit_dmodel_mmaps_cuda_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_omp_object

# Build rule for target.
gbkfit_dmodel_mmaps_omp_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_omp_object
.PHONY : gbkfit_dmodel_mmaps_omp_object

# fast build rule for target.
gbkfit_dmodel_mmaps_omp_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_object.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_object.dir/build
.PHONY : gbkfit_dmodel_mmaps_omp_object/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_omp_shared

# Build rule for target.
gbkfit_dmodel_mmaps_omp_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_omp_shared
.PHONY : gbkfit_dmodel_mmaps_omp_shared

# fast build rule for target.
gbkfit_dmodel_mmaps_omp_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_shared.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_shared.dir/build
.PHONY : gbkfit_dmodel_mmaps_omp_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_dmodel_mmaps_omp_static

# Build rule for target.
gbkfit_dmodel_mmaps_omp_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_dmodel_mmaps_omp_static
.PHONY : gbkfit_dmodel_mmaps_omp_static

# fast build rule for target.
gbkfit_dmodel_mmaps_omp_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/build.make src/gbkfit/gbkfit_dmodel_mmaps_omp/CMakeFiles/gbkfit_dmodel_mmaps_omp_static.dir/build
.PHONY : gbkfit_dmodel_mmaps_omp_static/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_object

# Build rule for target.
gbkfit_gmodel_gmodel1_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_object
.PHONY : gbkfit_gmodel_gmodel1_object

# fast build rule for target.
gbkfit_gmodel_gmodel1_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1/CMakeFiles/gbkfit_gmodel_gmodel1_object.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1/CMakeFiles/gbkfit_gmodel_gmodel1_object.dir/build
.PHONY : gbkfit_gmodel_gmodel1_object/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_static

# Build rule for target.
gbkfit_gmodel_gmodel1_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_static
.PHONY : gbkfit_gmodel_gmodel1_static

# fast build rule for target.
gbkfit_gmodel_gmodel1_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1/CMakeFiles/gbkfit_gmodel_gmodel1_static.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1/CMakeFiles/gbkfit_gmodel_gmodel1_static.dir/build
.PHONY : gbkfit_gmodel_gmodel1_static/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_shared

# Build rule for target.
gbkfit_gmodel_gmodel1_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_shared
.PHONY : gbkfit_gmodel_gmodel1_shared

# fast build rule for target.
gbkfit_gmodel_gmodel1_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1/CMakeFiles/gbkfit_gmodel_gmodel1_shared.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1/CMakeFiles/gbkfit_gmodel_gmodel1_shared.dir/build
.PHONY : gbkfit_gmodel_gmodel1_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_cuda_object

# Build rule for target.
gbkfit_gmodel_gmodel1_cuda_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_cuda_object
.PHONY : gbkfit_gmodel_gmodel1_cuda_object

# fast build rule for target.
gbkfit_gmodel_gmodel1_cuda_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_object.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_object.dir/build
.PHONY : gbkfit_gmodel_gmodel1_cuda_object/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_cuda_static

# Build rule for target.
gbkfit_gmodel_gmodel1_cuda_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_cuda_static
.PHONY : gbkfit_gmodel_gmodel1_cuda_static

# fast build rule for target.
gbkfit_gmodel_gmodel1_cuda_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_static.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_static.dir/build
.PHONY : gbkfit_gmodel_gmodel1_cuda_static/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_cuda_nvcc_shared

# Build rule for target.
gbkfit_gmodel_gmodel1_cuda_nvcc_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_cuda_nvcc_shared
.PHONY : gbkfit_gmodel_gmodel1_cuda_nvcc_shared

# fast build rule for target.
gbkfit_gmodel_gmodel1_cuda_nvcc_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_nvcc_shared.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_nvcc_shared.dir/build
.PHONY : gbkfit_gmodel_gmodel1_cuda_nvcc_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_cuda_nvcc_static

# Build rule for target.
gbkfit_gmodel_gmodel1_cuda_nvcc_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_cuda_nvcc_static
.PHONY : gbkfit_gmodel_gmodel1_cuda_nvcc_static

# fast build rule for target.
gbkfit_gmodel_gmodel1_cuda_nvcc_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_nvcc_static.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_nvcc_static.dir/build
.PHONY : gbkfit_gmodel_gmodel1_cuda_nvcc_static/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_cuda_shared

# Build rule for target.
gbkfit_gmodel_gmodel1_cuda_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_cuda_shared
.PHONY : gbkfit_gmodel_gmodel1_cuda_shared

# fast build rule for target.
gbkfit_gmodel_gmodel1_cuda_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_shared.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1_cuda/CMakeFiles/gbkfit_gmodel_gmodel1_cuda_shared.dir/build
.PHONY : gbkfit_gmodel_gmodel1_cuda_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_omp_shared

# Build rule for target.
gbkfit_gmodel_gmodel1_omp_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_omp_shared
.PHONY : gbkfit_gmodel_gmodel1_omp_shared

# fast build rule for target.
gbkfit_gmodel_gmodel1_omp_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1_omp/CMakeFiles/gbkfit_gmodel_gmodel1_omp_shared.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1_omp/CMakeFiles/gbkfit_gmodel_gmodel1_omp_shared.dir/build
.PHONY : gbkfit_gmodel_gmodel1_omp_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_omp_static

# Build rule for target.
gbkfit_gmodel_gmodel1_omp_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_omp_static
.PHONY : gbkfit_gmodel_gmodel1_omp_static

# fast build rule for target.
gbkfit_gmodel_gmodel1_omp_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1_omp/CMakeFiles/gbkfit_gmodel_gmodel1_omp_static.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1_omp/CMakeFiles/gbkfit_gmodel_gmodel1_omp_static.dir/build
.PHONY : gbkfit_gmodel_gmodel1_omp_static/fast

#=============================================================================
# Target rules for targets named gbkfit_gmodel_gmodel1_omp_object

# Build rule for target.
gbkfit_gmodel_gmodel1_omp_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_gmodel_gmodel1_omp_object
.PHONY : gbkfit_gmodel_gmodel1_omp_object

# fast build rule for target.
gbkfit_gmodel_gmodel1_omp_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_gmodel_gmodel1_omp/CMakeFiles/gbkfit_gmodel_gmodel1_omp_object.dir/build.make src/gbkfit/gbkfit_gmodel_gmodel1_omp/CMakeFiles/gbkfit_gmodel_gmodel1_omp_object.dir/build
.PHONY : gbkfit_gmodel_gmodel1_omp_object/fast

#=============================================================================
# Target rules for targets named gbkfit_fitter_mpfit_object

# Build rule for target.
gbkfit_fitter_mpfit_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_fitter_mpfit_object
.PHONY : gbkfit_fitter_mpfit_object

# fast build rule for target.
gbkfit_fitter_mpfit_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_fitter_mpfit/CMakeFiles/gbkfit_fitter_mpfit_object.dir/build.make src/gbkfit/gbkfit_fitter_mpfit/CMakeFiles/gbkfit_fitter_mpfit_object.dir/build
.PHONY : gbkfit_fitter_mpfit_object/fast

#=============================================================================
# Target rules for targets named gbkfit_fitter_mpfit_static

# Build rule for target.
gbkfit_fitter_mpfit_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_fitter_mpfit_static
.PHONY : gbkfit_fitter_mpfit_static

# fast build rule for target.
gbkfit_fitter_mpfit_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_fitter_mpfit/CMakeFiles/gbkfit_fitter_mpfit_static.dir/build.make src/gbkfit/gbkfit_fitter_mpfit/CMakeFiles/gbkfit_fitter_mpfit_static.dir/build
.PHONY : gbkfit_fitter_mpfit_static/fast

#=============================================================================
# Target rules for targets named gbkfit_fitter_mpfit_shared

# Build rule for target.
gbkfit_fitter_mpfit_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_fitter_mpfit_shared
.PHONY : gbkfit_fitter_mpfit_shared

# fast build rule for target.
gbkfit_fitter_mpfit_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_fitter_mpfit/CMakeFiles/gbkfit_fitter_mpfit_shared.dir/build.make src/gbkfit/gbkfit_fitter_mpfit/CMakeFiles/gbkfit_fitter_mpfit_shared.dir/build
.PHONY : gbkfit_fitter_mpfit_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_fitter_multinest_object

# Build rule for target.
gbkfit_fitter_multinest_object: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_fitter_multinest_object
.PHONY : gbkfit_fitter_multinest_object

# fast build rule for target.
gbkfit_fitter_multinest_object/fast:
	$(MAKE) -f src/gbkfit/gbkfit_fitter_multinest/CMakeFiles/gbkfit_fitter_multinest_object.dir/build.make src/gbkfit/gbkfit_fitter_multinest/CMakeFiles/gbkfit_fitter_multinest_object.dir/build
.PHONY : gbkfit_fitter_multinest_object/fast

#=============================================================================
# Target rules for targets named gbkfit_fitter_multinest_static

# Build rule for target.
gbkfit_fitter_multinest_static: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_fitter_multinest_static
.PHONY : gbkfit_fitter_multinest_static

# fast build rule for target.
gbkfit_fitter_multinest_static/fast:
	$(MAKE) -f src/gbkfit/gbkfit_fitter_multinest/CMakeFiles/gbkfit_fitter_multinest_static.dir/build.make src/gbkfit/gbkfit_fitter_multinest/CMakeFiles/gbkfit_fitter_multinest_static.dir/build
.PHONY : gbkfit_fitter_multinest_static/fast

#=============================================================================
# Target rules for targets named gbkfit_fitter_multinest_shared

# Build rule for target.
gbkfit_fitter_multinest_shared: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_fitter_multinest_shared
.PHONY : gbkfit_fitter_multinest_shared

# fast build rule for target.
gbkfit_fitter_multinest_shared/fast:
	$(MAKE) -f src/gbkfit/gbkfit_fitter_multinest/CMakeFiles/gbkfit_fitter_multinest_shared.dir/build.make src/gbkfit/gbkfit_fitter_multinest/CMakeFiles/gbkfit_fitter_multinest_shared.dir/build
.PHONY : gbkfit_fitter_multinest_shared/fast

#=============================================================================
# Target rules for targets named gbkfit_app_cli

# Build rule for target.
gbkfit_app_cli: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gbkfit_app_cli
.PHONY : gbkfit_app_cli

# fast build rule for target.
gbkfit_app_cli/fast:
	$(MAKE) -f src/apps/gbkfit_app_cli/CMakeFiles/gbkfit_app_cli.dir/build.make src/apps/gbkfit_app_cli/CMakeFiles/gbkfit_app_cli.dir/build
.PHONY : gbkfit_app_cli/fast

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... install"
	@echo "... list_install_components"
	@echo "... rebuild_cache"
	@echo "... uninstall"
	@echo "... install/strip"
	@echo "... install/local"
	@echo "... edit_cache"
	@echo "... gbkfit_object"
	@echo "... gbkfit_shared"
	@echo "... gbkfit_static"
	@echo "... gbkfit_cuda_object"
	@echo "... gbkfit_cuda_static"
	@echo "... gbkfit_cuda_shared"
	@echo "... gbkfit_fftw3_object"
	@echo "... gbkfit_fftw3_shared"
	@echo "... gbkfit_fftw3_static"
	@echo "... gbkfit_dmodel_scube_static"
	@echo "... gbkfit_dmodel_scube_object"
	@echo "... gbkfit_dmodel_scube_shared"
	@echo "... gbkfit_dmodel_scube_cuda_object"
	@echo "... gbkfit_dmodel_scube_cuda_static"
	@echo "... gbkfit_dmodel_scube_cuda_nvcc_shared"
	@echo "... gbkfit_dmodel_scube_cuda_shared"
	@echo "... gbkfit_dmodel_scube_cuda_nvcc_static"
	@echo "... gbkfit_dmodel_scube_omp_static"
	@echo "... gbkfit_dmodel_scube_omp_object"
	@echo "... gbkfit_dmodel_scube_omp_shared"
	@echo "... gbkfit_dmodel_mmaps_object"
	@echo "... gbkfit_dmodel_mmaps_static"
	@echo "... gbkfit_dmodel_mmaps_shared"
	@echo "... gbkfit_dmodel_mmaps_cuda_static"
	@echo "... gbkfit_dmodel_mmaps_cuda_nvcc_static"
	@echo "... gbkfit_dmodel_mmaps_cuda_object"
	@echo "... gbkfit_dmodel_mmaps_cuda_nvcc_shared"
	@echo "... gbkfit_dmodel_mmaps_cuda_shared"
	@echo "... gbkfit_dmodel_mmaps_omp_object"
	@echo "... gbkfit_dmodel_mmaps_omp_shared"
	@echo "... gbkfit_dmodel_mmaps_omp_static"
	@echo "... gbkfit_gmodel_gmodel1_object"
	@echo "... gbkfit_gmodel_gmodel1_static"
	@echo "... gbkfit_gmodel_gmodel1_shared"
	@echo "... gbkfit_gmodel_gmodel1_cuda_object"
	@echo "... gbkfit_gmodel_gmodel1_cuda_static"
	@echo "... gbkfit_gmodel_gmodel1_cuda_nvcc_shared"
	@echo "... gbkfit_gmodel_gmodel1_cuda_nvcc_static"
	@echo "... gbkfit_gmodel_gmodel1_cuda_shared"
	@echo "... gbkfit_gmodel_gmodel1_omp_shared"
	@echo "... gbkfit_gmodel_gmodel1_omp_static"
	@echo "... gbkfit_gmodel_gmodel1_omp_object"
	@echo "... gbkfit_fitter_mpfit_object"
	@echo "... gbkfit_fitter_mpfit_static"
	@echo "... gbkfit_fitter_mpfit_shared"
	@echo "... gbkfit_fitter_multinest_object"
	@echo "... gbkfit_fitter_multinest_static"
	@echo "... gbkfit_fitter_multinest_shared"
	@echo "... gbkfit_app_cli"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

