cmake_minimum_required(VERSION 3.18)
project(lxfu LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(_libtorch_root "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
set(_libtorch_nested "${_libtorch_root}/libtorch")
set(_libtorch_candidates ${_libtorch_root} ${_libtorch_nested})
set(_libtorch_selected "")

foreach(_libtorch_candidate IN LISTS _libtorch_candidates)
  set(_torch_config "${_libtorch_candidate}/share/cmake/Torch/TorchConfig.cmake")
  if(EXISTS "${_torch_config}")
    set(_libtorch_selected "${_libtorch_candidate}")
    break()
  endif()
endforeach()

if(_libtorch_selected)
  set(_libtorch_new_cfg "${_libtorch_selected}/share/cmake/Torch")
  set(_libtorch_old_cfg "${_libtorch_nested}/share/cmake/Torch")

  if(NOT Torch_DIR OR Torch_DIR STREQUAL "${_libtorch_old_cfg}")
    set(Torch_DIR "${_libtorch_new_cfg}" CACHE PATH "Torch directory" FORCE)
  endif()

  if(NOT DEFINED CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "${_libtorch_nested}")
    set(CMAKE_PREFIX_PATH "${_libtorch_selected}" CACHE STRING "CMake prefix search path" FORCE)
  endif()

  foreach(_torch_cache_var TORCH_LIBRARY c10_LIBRARY kineto_LIBRARY)
    if(DEFINED ${_torch_cache_var} AND ${_torch_cache_var} MATCHES "/libtorch/libtorch/")
      unset(${_torch_cache_var} CACHE)
    endif()
  endforeach()

  if(DEFINED FIND_PACKAGE_MESSAGE_DETAILS_Torch AND FIND_PACKAGE_MESSAGE_DETAILS_Torch MATCHES "/libtorch/libtorch/")
    unset(FIND_PACKAGE_MESSAGE_DETAILS_Torch CACHE)
  endif()
endif()

unset(_torch_cache_var)
unset(_libtorch_new_cfg)
unset(_libtorch_old_cfg)
unset(_libtorch_selected)
unset(_torch_config)
unset(_libtorch_candidate)
unset(_libtorch_candidates)
unset(_libtorch_nested)
unset(_libtorch_root)

find_package(Torch REQUIRED)

# Use system OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui objdetect)

find_library(PAM_LIB pam REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add FAISS subdirectory (CPU only)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable GPU support")
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Python bindings")
set(FAISS_OPT_LEVEL generic CACHE STRING "Use generic optimization")
set(BUILD_TESTING OFF CACHE BOOL "Disable FAISS tests")
add_subdirectory(faiss EXCLUDE_FROM_ALL)

# Build LMDB as a static library
add_library(lmdb STATIC 
    lmdb/mdb.c
    lmdb/midl.c
)
target_include_directories(lmdb PUBLIC lmdb)
set_target_properties(lmdb PROPERTIES LINKER_LANGUAGE C)

# Main LXFU executable
add_executable(lxfu src/lxfu.cpp)
target_link_libraries(lxfu PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS} faiss lmdb pthread)
target_include_directories(lxfu PRIVATE src)
target_compile_features(lxfu PRIVATE cxx_std_17)

# Original demo (optional)
add_executable(dinov3_demo src/main.cpp)
target_link_libraries(dinov3_demo PRIVATE "${TORCH_LIBRARIES}" ${OpenCV_LIBS} faiss lmdb pthread)
target_compile_features(dinov3_demo PRIVATE cxx_std_17)

add_library(pam_lxfu MODULE src/pam_lxfu.cpp)
target_link_libraries(pam_lxfu PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS} faiss lmdb pthread ${PAM_LIB})
target_include_directories(pam_lxfu PRIVATE src)
target_compile_features(pam_lxfu PRIVATE cxx_std_17)
set_target_properties(pam_lxfu PROPERTIES PREFIX "" SUFFIX ".so")

set_property(TARGET lxfu PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_property(TARGET dinov3_demo PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(LXFU_INSTALL_LIB_SUBDIR "${CMAKE_INSTALL_LIBDIR}/lxfu")
set(LXFU_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/lxfu")

foreach(target lxfu pam_lxfu)
  set_target_properties(${target} PROPERTIES
    INSTALL_RPATH "${LXFU_INSTALL_RPATH}"
  )
endforeach()

set(_torch_candidate_dirs
  "${CMAKE_SOURCE_DIR}/libtorch/lib"
  "${CMAKE_SOURCE_DIR}/libtorch/libtorch/lib"
)
foreach(_candidate IN LISTS _torch_candidate_dirs)
  if(EXISTS "${_candidate}")
    install(DIRECTORY "${_candidate}/"
            DESTINATION "${LXFU_INSTALL_LIB_SUBDIR}"
            FILES_MATCHING
            PATTERN "*.so"
            PATTERN "*.so.*"
            PATTERN "*.dylib"
            PATTERN "*.dll")
    break()
  endif()
endforeach()

# Libtorch binaries are all built with position independent code already.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Install rules
install(TARGETS lxfu DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(TARGETS pam_lxfu DESTINATION "${CMAKE_INSTALL_LIBDIR}/security")
install(FILES lxfu.conf DESTINATION "${CMAKE_INSTALL_SYSCONFDIR}/lxfu")
install(FILES dino.pt DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/lxfu")
