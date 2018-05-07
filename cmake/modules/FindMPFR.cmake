# Try to find the MPFR libraries
# MPFR_FOUND - system has MPFR lib
# MPFR_INCLUDE_DIR - the MPFR include directory
# MPFR_LIBRARIES_DIR - Directory where the MPFR libraries are located
# MPFR_LIBRARIES - the MPFR libraries
# MPFR_IN_CGAL_AUXILIARY - TRUE if the MPFR found is the one distributed with CGAL in the auxiliary folder

# TODO: support MacOSX

include(FindPackageHandleStandardArgs)


if(MPFR_INCLUDE_DIR)
  set(MPFR_in_cache TRUE)
else()
  set(MPFR_in_cache FALSE)
endif()
if(NOT MPFR_LIBRARIES)
  set(MPFR_in_cache FALSE)
endif()

# Is it already configured?
if (MPFR_in_cache)

  set(MPFR_FOUND TRUE)

else()

  find_path(MPFR_INCLUDE_DIR
    NAMES mpfr.h
    HINTS ENV MPFR_INC_DIR ENV MODULE_mpfr_ROOT_DIR
    ENV MPFR_DIR
    /usr/include
    PATH_SUFFIXES include
    DOC "The directory containing the MPFR header files"
    )


  find_library(MPFR_LIBRARIES NAMES mpfr
    HINTS ENV MPFR_LIB_DIR ENV MODULE_mpfr_ROOT_DIR
    ENV MPFR_DIR
    /usr/lib
    PATH_SUFFIXES lib lib64
    DOC "Path to the MPFR library"
    )

  if ( MPFR_LIBRARIES )
    get_filename_component(MPFR_LIBRARIES_DIR ${MPFR_LIBRARIES} PATH CACHE )
  endif()

  # Attempt to load a user-defined configuration for MPFR if couldn't be found
  if ( NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARIES_DIR )
    include( MPFRConfig OPTIONAL )
  endif()

  find_package_handle_standard_args(MPFR "DEFAULT_MSG" MPFR_LIBRARIES MPFR_INCLUDE_DIR)

endif()
