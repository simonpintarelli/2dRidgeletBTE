cmake_minimum_required (VERSION 2.8.8)

project(FFRT)

cmake_policy(SET CMP0046 NEW)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake/modules/")

set(USE_OMP                      ON CACHE BOOL "use OpenMP")
set(BUILD_TESTS                  NO CACHE BOOL "build tests")
set(ENABLE_PYTHON_BINDINGS       ON CACHE BOOL "use python")

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release"
    FORCE )
endif()

include(CheckCXXSourceCompiles)
include(cmake/checks/CMakeLists.txt)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

if(HAVE_CXX14_RETURN_AUTO)
  add_compile_options("-std=c++14")
  add_definitions("-DUSE_CXX14")
else()
  add_compile_options("-std=c++11")
endif()
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG" )
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -ftree-vectorize -mavx -O3 -DNDEBUG" )
SET(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3 -ftree-vectorize -ftree-vectorizer-verbose=2 -msse -msse2 -DNDEBUG " )
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -ftree-vectorize -ftree-vectorizer-verbose=2 -msse -msse2  " )

add_compile_options("-Wall")

set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")
set(EIGEN3_UNS_INCLUDE_DIR "$ENV{EIGEN3_UNSUPPORTED_INCLUDE_DIR}")

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(HDF5 REQUIRED)
FIND_PACKAGE(Boost 1.63 REQUIRED
  COMPONENTS
  program_options
  regex
  filesystem
  system)
FIND_PACKAGE(MPFR REQUIRED)
FIND_PACKAGE(FFTW3 REQUIRED)
FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(GTest)

if(ENABLE_PYTHON_BINDINGS)
  FIND_PACKAGE(PythonInterp REQUIRED)
  FIND_PACKAGE(PythonLibs 3 REQUIRED)
endif()

if(OPENMP_FOUND AND USE_OMP)
  add_definitions(-DUSE_OMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

exec_program(
  "git"
  ${CMAKE_CURRENT_SOURCE_DIR}
  ARGS "describe --always"
  OUTPUT_VARIABLE VERSION
  RETURN_VALUE GIT_QUERY_VERSION)
exec_program(
  "git"
  ${CMAKE_CURRENT_SOURCE_DIR}
  ARGS "describe --all"
  OUTPUT_VARIABLE BRANCHNAME
  RETURN_VALUE GIT_QUERY_BRANCH)
exec_program(
  "git"
  ${CMAKE_CURRENT_SOURCE_DIR}
  ARGS "describe --tags"
  OUTPUT_VARIABLE GIT_DESCRIBE_TAGS
  RETURN_VALUE GIT_QUERY_DT)
exec_program(
  "git diff | wc -l"
  ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GITDIFFC
  )
if(GITDIFFC EQUAL 0)
else()
  message("!!!! WARNING: source contains uncommited changes !!!!")
  set(BRANCHNAME "${BRANCHNAME}@@DIRTY@@")
endif()
if((GIT_QUERY_VERSION EQUAL 0) OR (GIT_QUERY_BRANCH EQUAL 0))
  string( REGEX MATCH "-g.*$" VERSION_SHA1 ${VERSION} )
  if(VERSION_SHA1 STREQUAL "")
    set(VERSION_SHA1 "${VERSION}")
  else()
    string( REGEX REPLACE "[-g]" "" VERSION_SHA1 ${VERSION_SHA1} )
  endif()
  message("version id: ${VERSION_SHA1}")
  message("branchname: ${BRANCHNAME}")
  set(GIT_SHA1 "${VERSION_SHA1}" )
  set(GIT_BNAME "${BRANCHNAME}")
  set(GIT_DT "${GIT_DESCRIBE_TAGS}")
else()
  message("!!!! WARNING: No git repo found! !!!!")
endif()
configure_file (
  "${PROJECT_SOURCE_DIR}/brt_config.h.in"
  "${PROJECT_BINARY_DIR}/brt_config.h"
  )

## annoying problem on Euler: HDF5_C_LIBRARIES contains HDF5_sz_NOTFOUND
set(HDF5_C_LIBS "${HDF5_C_LIBRARIES}")
set(HDF5_C_LIBRARIES "")
foreach(ii ${HDF5_C_LIBS})
  string(REGEX MATCHALL "NOTFOUND" oo ${ii})
  if(oo STREQUAL "")
    set(HDF5_C_LIBRARIES "${HDF5_C_LIBRARIES};${ii}")
  endif()
endforeach(ii)

INCLUDE_DIRECTORIES(BEFORE SYSTEM "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES(BEFORE SYSTEM "${Boost_INCLUDE_DIR}")
INCLUDE_DIRECTORIES(BEFORE SYSTEM "${EIGEN3_INCLUDE_DIR}")
INCLUDE_DIRECTORIES(BEFORE SYSTEM "${EIGEN3_INCLUDE_DIR}/unsupported")
INCLUDE_DIRECTORIES(BEFORE SYSTEM "${HDF5_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(BEFORE SYSTEM "${MPFR_INCLUDE_DIR}")
set(SYSTEM_LIBRARIES "${SYSTEM_LIBRARIES};${Boost_LIBRARIES};${HDF5_C_LIBRARIES};${HDF5_HL_LIBRARIES};${FFTW_LIBRARIES};${MPFR_LIBRARIES}")
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}")


# libraries
add_subdirectory(spectral/quadrature)
set(quad_location $<TARGET_FILE:quad>)
add_subdirectory(spectral/collision)
set(collision_location $<TARGET_FILE:collision>)
add_subdirectory(export)
set(export_location $<TARGET_FILE:export>)

# binaries
add_subdirectory(applications/mpi)
add_subdirectory(applications/python)
add_subdirectory(applications/post_processing)
add_subdirectory(applications/matrices)

if(GTEST_FOUND)
  add_subdirectory(test/gtest)
endif()

if(BUILD_TESTS)
  add_subdirectory(test)
  add_subdirectory(test/base)
  add_subdirectory(test/phase_space)
  add_subdirectory(test/mpi)
endif()
add_subdirectory(doc)

INSTALL(DIRECTORY base doc enum export fft operators ridgelet solver spectral traits DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")
