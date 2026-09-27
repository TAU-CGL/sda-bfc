cmake_minimum_required(VERSION 3.16)
project(SDA-BFC LANGUAGES CXX)

find_package(le3 REQUIRED)

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

add_executable(viz src/main.cpp)
target_link_libraries(viz PRIVATE le3::le3 ${frameworks})