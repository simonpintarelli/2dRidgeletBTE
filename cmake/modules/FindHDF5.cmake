#
# Find HDF5 C/ HL library
#
#
# HDF5_C_LIBRARIES  - hdf5 libaries
#
# HDF5_FOUND      - true when found


#check if HDF5_ROOT is defined in envir and use it
if (NOT HDF5_ROOT AND NOT $ENV{HDF5_ROOT} STREQUAL "")
  set (HDF5_ROOT $ENV{HDF5_ROOT})
endif ()

#check if HDF5ROOT is defined in envir and use it
if (NOT HDF5_ROOT AND NOT $ENV{HDF5ROOT} STREQUAL "")
  set (HDF5_ROOT $ENV{HDF5ROOT})
endif ()


# If HDF5_LIBRARYDIR was defined in the environment, use it.
IF( NOT $ENV{HDF5_LIBRARYDIR} STREQUAL "" )
  set(HDF5_LIBRARYDIR $ENV{HDF5_LIBRARYDIR})
ENDIF( NOT $ENV{HDF5_LIBRARYDIR} STREQUAL "" )

# figure out what the next statement does
IF( HDF5_ROOT )
  file(TO_CMAKE_PATH ${HDF5_ROOT} HDF5_ROOT)
ENDIF( HDF5_ROOT )


# set (HDF5_LIB_DIRS
#   ${HDF5_ROOT}
#   /usr/lib64/
#   )

set (HDF5_INC_DIRS
  ${HDF5_ROOT}
  ${HDF5_ROOT}/Include
  ${HDF5_ROOT}/include
  /usr/local/include
  /usr/include
  )

foreach (LIB hdf5 hdf5_hl)
  find_library(LOC_${LIB}
    NAMES
    ${LIB}
    HINTS
    ENV HDF5_ROOT
    PATH_SUFFIXES lib Lib lib64 Lib64
    )
  list(APPEND HDF5_C_LIBRARIES ${LOC_${LIB}})
endforeach()

find_path(HDF5_INCLUDE_DIRS hdf5.h HDF5_INC_DIRS)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HDF5 DEFAULT_MSG  HDF5_C_LIBRARIES)

mark_as_advanced(HDF5_C_LIBRARIES)
mark_as_advanced(HDF5_INCLUDE_DIRS)
