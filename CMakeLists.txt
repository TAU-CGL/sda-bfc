cmake_minimum_required(VERSION 3.16)

project(SDA-BFC LANGUAGES CXX)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

########################################
# Setup bootstrap macros and files
########################################
set(DEFAULT_PROJECT "resources")
set(DEFAULT_PROJECT_INITIAL_SCENE "two_arms.lua")
set(DEFAULT_PROJECT_ABSOLUTE "${CMAKE_SOURCE_DIR}/${DEFAULT_PROJECT}")

set(BOOTSTRAP_FILE "bootstrap.lua")
set(BOOTSTRAP_INPATH "${CMAKE_SOURCE_DIR}/${BOOTSTRAP_FILE}")
set(BOOTSTRAP_OUTPATH "${CMAKE_SOURCE_DIR}/bin/${BOOTSTRAP_FILE}")
add_compile_definitions(BOOTSTRAP_FILE="${BOOTSTRAP_FILE}")
if (EXISTS ${BOOTSTRAP_INPATH})
    configure_file(${BOOTSTRAP_INPATH} ${BOOTSTRAP_OUTPATH}  COPYONLY)
else()
    set(BOOSTRAP_CONTENT "\n\
        LE3GameConfig = {}\n\
        LE3GameConfig.ProjectPath = \"${DEFAULT_PROJECT_ABSOLUTE}\"\n\
        LE3GameConfig.InitialScene = \"${DEFAULT_PROJECT_INITIAL_SCENE}\"\n\
    ")
    string(REGEX REPLACE "\n[ \t]+" "\n" BOOSTRAP_CONTENT ${BOOSTRAP_CONTENT}) # Remove leading tabs after every newline
    string(REGEX REPLACE "^\n" "" BOOSTRAP_CONTENT ${BOOSTRAP_CONTENT}) # Remove the first empty line
    string(REGEX REPLACE "\n$" "" BOOSTRAP_CONTENT ${BOOSTRAP_CONTENT}) # Remove the last empty line
    file(CONFIGURE OUTPUT ${BOOTSTRAP_OUTPATH} CONTENT ${BOOSTRAP_CONTENT})
endif()

########################################

find_package(le3 REQUIRED)

########################################

# Link against CoreFoundation on macOS, and gtk on Linux, for native_message_box
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    find_library(corefoundation_lib CoreFoundation)
    set(frameworks ${corefoundation_lib})
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    find_package(PkgConfig REQUIRED)
    PKG_CHECK_MODULES( GTK REQUIRED gtk+-3.0 )
    INCLUDE_DIRECTORIES(${GTK_INCLUDE_DIRS})
    LINK_DIRECTORIES(${GTK_LIBRARY_DIRS})
    ADD_DEFINITIONS(${GTK_CFLAGS_OTHER})
    link_libraries( ${GTK_LIBRARIES} )
    set(frameworks  ${GTK_LIBRARIES})
else() # MSVC
    set(frameworks)
    string(APPEND CMAKE_C_FLAGS " /MP$ENV{MSVC_PARALLEL}")
    string(APPEND CMAKE_CXX_FLAGS " /MP$ENV{MSVC_PARALLEL}")
endif()

########################################
# dabfc: standalone library (no LightEngine3)
########################################

find_package(CGAL REQUIRED)
find_package(fcl REQUIRED)
add_library(dabfc STATIC
    src/dabfc/sdf.cpp
    src/dabfc/collision.cpp
    src/dabfc/robot_arm.cpp
    src/dabfc/ur5e.cpp
    src/dabfc/dual_arm_simulator.cpp
)
target_include_directories(dabfc PUBLIC include)
target_link_libraries(dabfc PUBLIC CGAL::CGAL glm::glm assimp::assimp fcl)
target_compile_features(dabfc PUBLIC cxx_std_20)

add_executable(viz
    src/main.cpp
    src/lua_dual_arm.cpp
)
target_link_libraries(viz PRIVATE dabfc le3::le3 ${frameworks})

########################################
# Tests
########################################

include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
enable_testing()

foreach(test test_sdf test_dual_arm)
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} PRIVATE dabfc GTest::gtest_main)
    target_compile_definitions(${test} PRIVATE MESH_DIR="${CMAKE_SOURCE_DIR}/resources/ur5e/visual")
    gtest_discover_tests(${test})
endforeach()

########################################
# Benchmarks (always optimized)
########################################

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/libomp) # brew install libomp
endif()
find_package(OpenMP REQUIRED)
add_executable(bench_sdf src/benchmarks/bench_sdf.cpp src/dabfc/sdf.cpp)
target_include_directories(bench_sdf PRIVATE include)
target_link_libraries(bench_sdf PRIVATE CGAL::CGAL glm::glm assimp::assimp OpenMP::OpenMP_CXX)
target_compile_features(bench_sdf PRIVATE cxx_std_20)
target_compile_options(bench_sdf PRIVATE -O2)
target_compile_definitions(bench_sdf PRIVATE NDEBUG CGAL_NDEBUG MESH_DIR="${CMAKE_SOURCE_DIR}/resources/ur5e/visual")

########################################
# Formatting: cmake --build build --target format
########################################

find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    file(GLOB_RECURSE FORMAT_SOURCES CONFIGURE_DEPENDS include/*.h src/*.cpp tests/*.cpp)
    add_custom_target(format COMMAND ${CLANG_FORMAT} -i ${FORMAT_SOURCES})
endif()
