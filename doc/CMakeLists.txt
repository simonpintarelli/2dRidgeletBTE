# Build documentation (requires doxygen 1.8.1 or later)
find_package(Doxygen)

if(DOXYGEN_FOUND)
  set(DOXYGEN_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/doxycfg")
  # We have to configure the Doxygen configuration file because the input
  # paths have to be adapted
  configure_file(doxycfg.in ${DOXYGEN_CONFIG_FILE} @ONLY IMMEDIATE)
  set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  # Add a target so that the documentation can be built with "make doc"
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG_FILE}
    WORKING_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
endif(DOXYGEN_FOUND)
