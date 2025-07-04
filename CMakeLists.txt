cmake_minimum_required(VERSION 3.14)

project(MathCV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(UnitTestCPP)

set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTORCC OFF)

file(GLOB SOURCES
  "src/LinearAlgebra/*.cpp"
  "src/LinearAlgebra/vector2d.cpp"
  "src/LinearAlgebra/matrix.cpp"
)

file(GLOB HEADERS
  "src/LinearAlgebra/*.h"
  "src/LinearAlgebra/vector2d.h"
  "src/LinearAlgebra/vectornd.h"
  "src/LinearAlgebra/matrix.h"
  "src/LinearAlgebra/math_enums.h"
)

add_library(MathCV SHARED
  ${SOURCES}
  ${HEADERS}
  MathCV_global.h
)

target_compile_definitions(MathCV PRIVATE MATHCV_LIBRARY)

# Тесты
enable_testing()
    
add_executable(MathCVTests
    MathCVTests.cpp
)

target_link_libraries(MathCVTests
    PRIVATE
    MathCV
    UnitTestCPP
)

add_test(NAME MathCVTests COMMAND MathCVTests)

# Добавляем пользовательскую цель для запуска тестов
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS MathCVTests
    COMMENT "Running tests..."
)