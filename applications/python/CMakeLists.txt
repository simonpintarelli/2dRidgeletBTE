# # Set list of source files
include_directories(${PROJECT_SOURCE_DIR}/src)
if(ENABLE_PYTHON_BINDINGS)
  INCLUDE_DIRECTORIES(BEFORE SYSTEM "${PYTHON_INCLUDE_DIR}")

  find_package(Boost COMPONENTS python3)
  set(Boost_PYTHON_LIB ${Boost_LIBRARIES})

  if(BOOST_NPY_FOUND)
    if(NOT _OFFICIAL_BOOST_NUMPY_FOUND)
      add_compile_options(-DUSE_DEPRECATED_BOOST_NPY)
    endif()

    set(libname pyFFRT)
    set(SOURCES pyFFRT.C)
    INCLUDE_DIRECTORIES(BEFORE SYSTEM "${BOOST_NPY_INCLUDE_DIR}")
    add_library(${libname} SHARED ${SOURCES})
    set_target_properties(
      ${libname}
      PROPERTIES
      COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DPYTHON -fPIC "
      LINK_FLAGS "-shared -fPIC"
      LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/pyFFRT )
    target_link_libraries(${libname} "${SYSTEM_LIBRARIES};${BOOST_NPY_LIBRARIES};${Boost_PYTHON_LIB}")

    configure_file (
      "${PROJECT_SOURCE_DIR}/applications/python/__init__.py.in"
      "${PROJECT_BINARY_DIR}/lib/pyFFRT/__init__.py"
      )

    INSTALL (TARGETS ${libname}
      LIBRARY DESTINATION  ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/pyFFRT )
    INSTALL(FILES "${PROJECT_BINARY_DIR}/lib/pyFFRT/__init__.py" DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/pyFFRT)
    FILE(GLOB pyfiles "*py")
    INSTALL(FILES ${pyfiles} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.6/site-packages/pyFFRT)
  endif()
endif()
