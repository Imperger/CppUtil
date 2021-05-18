cmake_minimum_required (VERSION 3.12)
project (CppUtilTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_compile_options(/utf-8)
elseif (UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif (UNIX)

include_directories(../include)

add_executable (CppUtilTest main.cpp)

if (UNIX)
    target_link_libraries(CppUtilTest PRIVATE Threads::Threads)
endif (UNIX)