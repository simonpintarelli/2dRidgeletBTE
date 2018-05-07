set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

INCLUDE_DIRECTORIES(BEFORE SYSTEM "${MPI_CXX_INCLUDE_PATH}")
set(SYSTEM_LIBRARIES "${MPI_CXX_LIBRARIES};${SYSTEM_LIBRARIES}")

set(EXE test_hdf_writer)
add_executable(${EXE} main_${EXE}.cpp)
add_dependencies(${EXE} quad collision export)
target_link_libraries(${EXE} "${quad_location};${collision_location};${export_location};${SYSTEM_LIBRARIES}")
target_compile_options(${EXE} PRIVATE -DUSE_MPI)
INSTALL(
  TARGETS ${EXE} RUNTIME
  DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

set(EXE test_output_handler)
add_executable(${EXE} main_${EXE}.cpp)
add_dependencies(${EXE} quad collision export)
target_link_libraries(${EXE} "${quad_location};${collision_location};${export_location};${SYSTEM_LIBRARIES}")
target_compile_options(${EXE} PRIVATE -DUSE_MPI)
INSTALL(
  TARGETS ${EXE} RUNTIME
  DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")


add_custom_command(TARGET ${EXE} POST_BUILD
                   COMMAND "cp" "-a"
                       ${CMAKE_SOURCE_DIR}/test/mpi/init_bte_omp.py $<TARGET_FILE_DIR:${EXE}>/ )
