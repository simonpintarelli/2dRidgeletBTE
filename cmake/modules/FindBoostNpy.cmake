# Try to find the boost-numpy libraries
# BOOST_NPY_FOUND - system has boost-numpy lib
# BOOST_NPY_INCLUDE_DIR - the boost-numpy include directory
# BOOST_NPY_LIBRARIES - the boost-numpy libraries
# BOOST_NPY_LIBRARIES_DIR - Directory where the boost-numpy libraries are located
# BOOST_DEPRECATED_NPY_VERSION - true if using boost-numpy from the inofficial boost pkg
# check first for numpy library in official boost

include(FindPackageHandleStandardArgs)

if(PYTHON_VERSION_STRING GREATER 3.0)
  message("using python 3: attempt to find corresponding Boost Python Numpy")
  set(_BNUMPYNAME boost_numpy)
  set(_NAME_OFFICIAL numpy3)
else()
  message("using python 2.7: attempt to find corresponding Boost Python Numpy")
  set(_BNUMPYNAME boost_numpy2.7)
  set(_NAME_OFFICIAL numpy)
endif()

find_package(Boost COMPONENTS ${_NAME_OFFICIAL})

if(Boost_NUMPY3_FOUND OR Boost_NUMPY_FOUND)
  message("FindBoostNpy (official boost): boost-libs=${Boost_LIBRARIES}")
  set(BOOST_NPY_LIBRARIES "${Boost_LIBRARIES}")
  set(BOOST_NPY_INCLUDE_DIR "${Boost_INCLUDE_DIR}")
  set(_OFFICIAL_BOOST_NUMPY_FOUND True)
endif()

if(NOT _OFFICIAL_BOOST_NUMPY_FOUND)
  find_path(BOOST_NPY_INCLUDE_DIR
    NAMES boost/numpy.hpp
    HINTS ENV BOOST_NPY_INC_DIR
    ENV BOOST_NPY_DIR
    /usr/include/
    /usr/local/include/
    PATH_SUFFIXES include
    DOC "The directory containing the BOOST_NPY header files"
    )
  find_library(BOOST_NPY_LIBRARIES NAMES ${_BNUMPYNAME} boost_numpy
    HINTS ENV BOOST_NPY_LIB_DIR
    ENV BOOST_NPY_DIR
    /usr/lib
    /usr/local/lib
    /usr/local/lib64
    PATH_SUFFIXES lib lib64
    DOC "Path to the BOOST_NPY library"
    )
endif()

if ( BOOST_NPY_LIBRARIES )
  get_filename_component(BOOST_NPY_LIBRARIES_DIR ${BOOST_NPY_LIBRARIES} DIRECTORY CACHE)
endif()

# workaround for cmake 3.11.1 (it wants to link against boost_python for v2 and 3)
set(BOOST_NPY_LIBRARIESw "")
foreach(ii ${BOOST_NPY_LIBRARIES})
  string(REGEX MATCHALL ".*libboost_python.so.*" oo ${ii})
  if(oo STREQUAL "")
    set(BOOST_NPY_LIBRARIESw "${BOOST_NPY_LIBRARIESw};${ii}")
  endif()
endforeach(ii)
set(BOOST_NPY_LIBRARIES ${BOOST_NPY_LIBRARIESw})
message("BOOST_NPY_LIBRARIES: ${BOOST_NPY_LIBRARIES}")


find_package_handle_standard_args(BOOST_NPY "DEFAULT_MSG" BOOST_NPY_LIBRARIES BOOST_NPY_INCLUDE_DIR)
