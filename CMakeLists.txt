cmake_minimum_required(VERSION 3.0)

project(statistical_outlier_removal LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "statistical_outlier_removal")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 

# .exe
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if (WIN32)
# -EHsc (enable proper Exxeption Handling) needs to avoid C4530
# -Wall is too noisy so that set -W4. 
# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level?view=vs-2017
# "However, for a new project, it may be best to use /W4 in all compilations;
# this will ensure the fewest possible hard-to-find code defects."
set(CMAKE_CXX_FLAGS "-std=c++14 -W4 -EHsc")
endif()

find_package(OpenCV REQUIRED)

add_executable(statistical_outlier_removal_run
    pc/main.cpp)

target_include_directories(statistical_outlier_removal_run PRIVATE include ${OpenCV_INCLUDE_DIRS})

target_link_libraries(statistical_outlier_removal_run
    ${OpenCV_LIBS}
    )