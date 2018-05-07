file(GLOB files RELATIVE ${CMAKE_SOURCE_DIR}/test "main*cpp")

foreach(file ${files})
  string(REGEX REPLACE "(main_|.cpp)" "" target ${file})
  add_executable(${target} ${file})
  add_dependencies(${target} quad)
  target_link_libraries(${target} "${SYSTEM_LIBRARIES};${quad_location}")
endforeach()
