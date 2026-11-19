cmake_minimum_required(VERSION 3.20)
project(pcl_eval_native LANGUAGES CXX)

include(CTest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 CONFIG REQUIRED)
find_package(PCL CONFIG REQUIRED COMPONENTS common io search kdtree features)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(primitive_core
    src/primitive_pipeline.cpp
)
target_include_directories(primitive_core PUBLIC include ${PCL_INCLUDE_DIRS})
target_link_libraries(primitive_core
    PUBLIC Eigen3::Eigen ${PCL_LIBRARIES} nlohmann_json::nlohmann_json
)
target_compile_definitions(primitive_core PUBLIC ${PCL_DEFINITIONS})
if(MSVC)
    target_compile_options(primitive_core PRIVATE /W4 /permissive- /EHsc /bigobj)
else()
    target_compile_options(primitive_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(native_primitive_fit src/main.cpp)
target_link_libraries(native_primitive_fit PRIVATE primitive_core)

if(BUILD_TESTING)
    add_executable(primitive_core_tests tests/test_core.cpp)
    target_link_libraries(primitive_core_tests PRIVATE primitive_core)
    add_test(NAME primitive_core_tests COMMAND primitive_core_tests)
endif()

