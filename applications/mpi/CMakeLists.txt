set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

INCLUDE_DIRECTORIES(BEFORE SYSTEM "${MPI_CXX_INCLUDE_PATH}")
set(SYSTEM_LIBRARIES "${MPI_CXX_LIBRARIES};${SYSTEM_LIBRARIES}")

file(GLOB files RELATIVE ${CMAKE_SOURCE_DIR}/applications/mpi "main*cpp")
foreach(file ${files})
  string(REGEX REPLACE "(main_|.cpp)" "" target ${file})
  add_executable(${target} ${file})
  add_dependencies(${target} quad collision export)
  target_link_libraries(${target} "${quad_location};${collision_location};${export_location};${SYSTEM_LIBRARIES}")
  SET_TARGET_PROPERTIES(${target} PROPERTIES
    COMPILE_DEFINITIONS "USE_MPI")
  INSTALL(
    TARGETS ${target} RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

endforeach()


set(EXE bte_omp_ftcg)
add_executable(${EXE} main_bte_omp.cpp)
add_dependencies(${EXE} quad collision export)
target_link_libraries(${EXE} "${quad_location};${collision_location};${export_location};${SYSTEM_LIBRARIES}")
target_compile_options(${EXE} PRIVATE -DUSE_MPI -DFTCG)
INSTALL(
  TARGETS ${EXE} RUNTIME
  DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
