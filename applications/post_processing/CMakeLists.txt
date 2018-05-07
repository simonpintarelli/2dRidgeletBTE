FIND_PACKAGE(YamlCpp)
# YAMLCPP_INCLUDE_DIR YAMLCPP_LIBRARY

set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
INCLUDE_DIRECTORIES(BEFORE SYSTEM "${MPI_CXX_INCLUDE_PATH}")
set(SYSTEM_LIBRARIES "${MPI_CXX_LIBRARIES};${SYSTEM_LIBRARIES}")

if(YAMLCPP_FOUND)
  include_directories(BEFORE SYSTEM ${YAMLCPP_INCLUDE_DIR})
  set(EXE active_coefficients)
  add_executable(${EXE} active_coefficients.cpp)
  add_dependencies(${EXE} quad collision)
  target_link_libraries(${EXE} "${quad_location};${collision_location};${SYSTEM_LIBRARIES};${YAMLCPP_LIBRARY}")
  INSTALL(
    TARGETS ${EXE} RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
else()
  message("Could not find YAML!")
endif()

set(EXE compute_rt)
add_executable(${EXE} compute_rt.cpp)
add_dependencies(${EXE} quad collision)
target_link_libraries(${EXE} "${quad_location};${collision_location};${SYSTEM_LIBRARIES};${YAMLCPP_LIBRARY}")
INSTALL(
  TARGETS ${EXE} RUNTIME
  DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
