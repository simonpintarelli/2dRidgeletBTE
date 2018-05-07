set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

file(GLOB files RELATIVE ${CMAKE_SOURCE_DIR}/applications/matrices "main*cpp")
foreach(file ${files})
  string(REGEX REPLACE "(main_|.cpp)" "" target ${file})
  add_executable(${target} ${file})
  add_dependencies(${target} quad export)
  target_link_libraries(${target} "${quad_location};${export_location};${SYSTEM_LIBRARIES}")
  INSTALL(
    TARGETS ${target} RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endforeach()
