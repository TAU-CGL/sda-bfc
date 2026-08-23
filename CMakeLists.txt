cmake_minimum_required(VERSION 3.20)

project(_sda_bfc)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(fmt REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

nanobind_add_module(_sda_bfc src/sda_bfc_py.cpp)
target_link_libraries(_sda_bfc PRIVATE fmt::fmt)

if (WIN32)
    install(TARGETS _sda_bfc LIBRARY DESTINATION lib/site-packages/sda_bfc)
else()
    install(TARGETS _sda_bfc LIBRARY DESTINATION lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/sda_bfc)
endif()
