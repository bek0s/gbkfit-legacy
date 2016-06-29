# ==============================================================================
#
#	CMake template file for uninstalling all installed files
#
# ==============================================================================

# Make sure install_manifest.txt exists
if (NOT EXISTS "/home/bekos/code/gbkfit/gbkfit/install_manifest.txt")
    message(FATAL_ERROR "Cannot find install manifest: \"/home/bekos/code/gbkfit/gbkfit/install_manifest.txt\"")
endif()

# Open install_manifest.txt for reading
file(READ "/home/bekos/code/gbkfit/gbkfit/install_manifest.txt" files)

# Make some ajustments to the filenames
string(REGEX REPLACE "\n" ";" files "${files}")
list(REVERSE files)

# Iterate through all the installed files and remove them
foreach (file ${files})
    message(STATUS "Uninstalling: $ENV{DESTDIR}${file}")
    if (EXISTS "$ENV{DESTDIR}${file}")
	execute_process(
	    COMMAND /usr/bin/cmake -E remove "$ENV{DESTDIR}${file}"
	    OUTPUT_VARIABLE rm_out
	    RESULT_VARIABLE rm_retval
	)
	if(NOT ${rm_retval} EQUAL 0)
            message(FATAL_ERROR "Problem while removing \"$ENV{DESTDIR}${file}\"")
	endif ()
    else ()
	message(STATUS "File \"$ENV{DESTDIR}${file}\" does not exist.")
    endif ()
endforeach(file)
