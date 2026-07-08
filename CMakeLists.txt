cmake_minimum_required(VERSION 3.15)

project(Aurelius)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

include_directories(include)

add_executable(Aurelius
    src/main.cpp
    src/Activation.cpp
    src/Dataset.cpp
    src/DenseLayer.cpp
    src/Loss.cpp
    src/Model.cpp
)

target_link_libraries(Aurelius PRIVATE Eigen3::Eigen)

target_compile_definitions(Aurelius PRIVATE
    PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

target_compile_options(Aurelius PRIVATE
    -O2
)