include_directories(${PROJECT_SOURCE_DIR}/src)
FILE(GLOB_RECURSE sources "*.cpp")
add_library(collision ${sources})

INSTALL(TARGETS collision ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
