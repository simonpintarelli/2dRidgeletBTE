include_directories(${PROJECT_SOURCE_DIR}/src)
FILE(GLOB sources "*.cpp")
add_library(quad ${sources})
target_link_libraries(quad "${MPFR_LIBRARIES}")

INSTALL(TARGETS quad ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
