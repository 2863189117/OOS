if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.24)
  project(oos_lxe_plugin LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  find_package(oos CONFIG REQUIRED)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  find_package(HDF5 1.14 REQUIRED COMPONENTS C)
  find_package(nlohmann_json 3.11 REQUIRED)
  option(OOS_LXE_ENABLE_CUDA "Build the dual-phase TPC LXe CUDA evaluator" OFF)
else()
  set(OOS_LXE_ENABLE_CUDA ${OOS_ENABLE_CUDA})
endif()

add_library(oos_lxe_plugin SHARED lxe_plugin.cpp)
target_include_directories(oos_lxe_plugin PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(oos_lxe_plugin PRIVATE
  oos::oos_core
  OpenMP::OpenMP_CXX
  nlohmann_json::nlohmann_json
  ${HDF5_LIBRARIES}
)
set_target_properties(oos_lxe_plugin PROPERTIES CXX_VISIBILITY_PRESET hidden)

if(OOS_LXE_ENABLE_CUDA)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
  endif()
  target_sources(oos_lxe_plugin PRIVATE lxe_function_cuda.cu)
  target_link_libraries(oos_lxe_plugin PRIVATE CUDA::cudart)
  target_compile_features(oos_lxe_plugin PRIVATE cuda_std_17)
  target_compile_definitions(oos_lxe_plugin PRIVATE OOS_LXE_HAS_CUDA=1)
endif()

include(GNUInstallDirs)
install(TARGETS oos_lxe_plugin
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
