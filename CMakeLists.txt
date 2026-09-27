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
find_package(Bullet REQUIRED) # headers only, for src/lua_physics_ext.cpp

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

add_executable(viz src/main.cpp src/lua_imgui_ext.cpp src/lua_physics_ext.cpp)
target_include_directories(viz PRIVATE ${BULLET_INCLUDE_DIRS})
target_link_libraries(viz PRIVATE le3::le3 ${frameworks})