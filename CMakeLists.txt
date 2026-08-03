cmake_minimum_required(VERSION 3.24)
project(OOS VERSION 0.1.0 LANGUAGES C CXX)

# Production installs use the locked workspace dependency prefix. Preserve
# non-system link directories in installed RPATHs so executables and plugins
# run without an activation shell or LD_LIBRARY_PATH.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/deps.lock.json" OOS_DEPS_LOCK_SHA256)
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE OOS_GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

option(OOS_ENABLE_CUDA "Build the native CUDA runtime" ON)
option(OOS_BUILD_TESTS "Build tests" ON)
option(OOS_BUILD_LXE_PLUGIN "Build the optional dual-phase TPC LXe plugin" OFF)
option(OOS_INSTALL_EXAMPLES "Install detector-specific examples" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(embree 4.3 REQUIRED)
find_package(HDF5 1.14 REQUIRED COMPONENTS C)
find_package(yaml-cpp 0.8 REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED)

add_library(oos_core
  src/analytic_geometry.cpp
  src/builder.cpp
  src/effective_response.cpp
  src/function_operator.cpp
  src/geometry.cpp
  src/hash.cpp
  src/hdf5_io.cpp
  src/physics.cpp
  src/plugin.cpp
  src/regression.cpp
  src/scene.cpp
  src/source.cpp
  src/surface_behavior.cpp
  src/solver.cpp
  src/validation.cpp
)
add_library(oos::oos_core ALIAS oos_core)
target_include_directories(oos_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(oos_core
  PUBLIC
    OpenMP::OpenMP_CXX
    embree
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    ${HDF5_LIBRARIES}
    ${CMAKE_DL_LIBS}
)
target_compile_options(oos_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
target_compile_definitions(oos_core PRIVATE
  OOS_DEPS_LOCK_SHA256="${OOS_DEPS_LOCK_SHA256}"
  OOS_GIT_COMMIT="${OOS_GIT_COMMIT}"
)

if(OOS_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(oos_cuda src/cuda_solver.cu)
  target_include_directories(oos_cuda PUBLIC include)
  target_link_libraries(oos_cuda PUBLIC oos_core CUDA::cudart CUDA::cusparse CUDA::cublas)
  target_compile_features(oos_cuda PUBLIC cuda_std_17)
  target_compile_definitions(oos_core PUBLIC OOS_HAS_CUDA=1)
endif()

add_executable(oos src/main.cpp)
target_link_libraries(oos PRIVATE oos_core)
if(OOS_ENABLE_CUDA)
  target_link_libraries(oos PRIVATE oos_cuda)
endif()

add_executable(oos-efficiency src/efficiency_main.cpp)
target_link_libraries(oos-efficiency PRIVATE oos_core)
if(OOS_ENABLE_CUDA)
  target_link_libraries(oos-efficiency PRIVATE oos_cuda)
endif()

add_executable(oos-regress src/regress_main.cpp)
target_link_libraries(oos-regress PRIVATE oos_core)
if(OOS_ENABLE_CUDA)
  target_link_libraries(oos-regress PRIVATE oos_cuda)
endif()

if(OOS_BUILD_LXE_PLUGIN)
  add_subdirectory(examples/dual-phase-tpc/plugin)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/oos-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/oos-config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/oos"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/oos-config-version.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)
install(TARGETS oos oos-efficiency oos-regress oos_core
  EXPORT OOSTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT OOSTargets
  FILE OOSTargets.cmake
  NAMESPACE oos::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/oos"
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/oos-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/oos-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/oos"
)
install(FILES deps.lock.json LICENSE PROVENANCE.md DESTINATION share/oos)
if(OOS_INSTALL_EXAMPLES)
  install(DIRECTORY examples/dual-phase-tpc
    DESTINATION share/oos/examples
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "tests" EXCLUDE
  )
  install(DIRECTORY examples/pet-4x4
    DESTINATION share/oos/examples
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "tests" EXCLUDE
  )
endif()

if(OOS_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3.7 REQUIRED)
  add_library(oos_test_local_plugin SHARED tests/local_plugin.cpp)
  target_include_directories(oos_test_local_plugin PRIVATE include)
  target_link_libraries(oos_test_local_plugin PRIVATE
    nlohmann_json::nlohmann_json
  )
  set_target_properties(oos_test_local_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
  )
  add_library(oos_test_nonlocal_plugin SHARED tests/nonlocal_plugin.cpp)
  target_include_directories(oos_test_nonlocal_plugin PRIVATE include)
  target_link_libraries(oos_test_nonlocal_plugin PRIVATE
    nlohmann_json::nlohmann_json
  )
  set_target_properties(oos_test_nonlocal_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
  )
  add_library(oos_test_function_operator SHARED
    tests/function_operator_plugin.cpp
  )
  target_include_directories(oos_test_function_operator PRIVATE include)
  set_target_properties(oos_test_function_operator PROPERTIES
    CXX_VISIBILITY_PRESET hidden
  )
  add_executable(oos_tests
    tests/test_geometry.cpp
    tests/test_builder.cpp
    tests/test_end_to_end.cpp
    tests/test_effective_response.cpp
    tests/test_function_operator.cpp
    tests/test_hash.cpp
    tests/test_hdf5.cpp
    tests/test_plugin.cpp
    tests/test_regression.cpp
    tests/test_physics.cpp
    tests/test_solver.cpp
    tests/test_source.cpp
    tests/test_surface_behavior.cpp
    tests/test_validation.cpp
  )
  target_link_libraries(oos_tests PRIVATE oos_core Catch2::Catch2WithMain)
  add_dependencies(oos_tests oos_test_local_plugin oos_test_nonlocal_plugin
    oos_test_function_operator)
  target_compile_definitions(oos_tests PRIVATE
    OOS_TEST_LOCAL_PLUGIN_PATH="$<TARGET_FILE:oos_test_local_plugin>"
    OOS_TEST_NONLOCAL_PLUGIN_PATH="$<TARGET_FILE:oos_test_nonlocal_plugin>"
    OOS_TEST_FUNCTION_OPERATOR_PATH="$<TARGET_FILE:oos_test_function_operator>"
  )
  if(OOS_BUILD_LXE_PLUGIN)
    add_dependencies(oos_tests oos_lxe_plugin)
    target_compile_definitions(oos_tests PRIVATE
      OOS_TEST_LXE_PLUGIN_PATH="$<TARGET_FILE:oos_lxe_plugin>"
    )
  endif()
  include(Catch)
  catch_discover_tests(oos_tests)
  if(OOS_ENABLE_CUDA)
    add_executable(oos_cuda_tests tests/test_cuda_solver.cpp)
    target_link_libraries(oos_cuda_tests PRIVATE oos_cuda Catch2::Catch2WithMain)
    if(OOS_BUILD_LXE_PLUGIN)
      add_dependencies(oos_cuda_tests oos_lxe_plugin)
      target_compile_definitions(oos_cuda_tests PRIVATE
        OOS_TEST_LXE_PLUGIN_PATH="$<TARGET_FILE:oos_lxe_plugin>"
      )
    endif()
    catch_discover_tests(oos_cuda_tests)
  endif()
endif()
