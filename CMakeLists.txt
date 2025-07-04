cmake_minimum_required(VERSION 3.14)
project(MathCV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(unittest-cpp)

file(GLOB SOURCES 
    "src/LinearAlgebra/*.cpp"
    "*.cpp"
)
file(GLOB HEADERS "src/LinearAlgebra/*.h")

add_library(MathCV SHARED ${SOURCES} ${HEADERS} MathCV_global.h)
target_compile_definitions(MathCV PRIVATE MATHCV_LIBRARY)

enable_testing()
add_executable(MathCVTests MathCVTests.cpp)
target_link_libraries(MathCVTests PRIVATE MathCV UnitTest++)
add_test(NAME MathCVTests COMMAND MathCVTests)

# Добавляем кастомную цель, которая зависит от сборки тестов
add_custom_target(run_tests ALL
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS MathCVTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests..."
)