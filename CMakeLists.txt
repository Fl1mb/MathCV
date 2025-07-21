cmake_minimum_required(VERSION 3.14)
project(MathCV LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(unittest-cpp)
find_package(CUDA REQUIRED)

file(GLOB SOURCES 
    "src/LinearAlgebra/*.cpp"
    "cuda/*.cpp"
    "*.cpp"
)
file(GLOB HEADERS 
    "src/LinearAlgebra/*.h"
    "cuda/*.h"
    )

add_library(MathCV SHARED ${SOURCES} ${HEADERS} MathCV_global.h)
target_compile_definitions(MathCV PRIVATE MATHCV_LIBRARY)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

option(${PROJECT_NAME}_BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" OFF)
if(${PROJECT_NAME}_BUILD_DOCUMENTATION)
	find_package (Doxygen)
    if(NOT DOXYGEN_FOUND)
	message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc ALL
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
	
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif(${PROJECT_NAME}_BUILD_DOCUMENTATION)

option(${PROJECT_NAME}_BUILD_TESTS "Build ${PROJECT_NAME} tests as separate executable" ON)
option(${PROJECT_NAME}_RUN_TESTS "Run ${PROJECT_NAME} tests" ON)
if (${PROJECT_NAME}_BUILD_TESTS)
    set (TESTS_FILES MathCVTests.cxx )

    enable_testing()
    add_definitions(-DINCLUDE_UNITTESTS)

    if(NOT TARGET UnitTest++)
        add_subdirectory(unittest-cpp ${CMAKE_CURRENT_BINARY_DIR}/UnitTest++)
    endif()

	add_executable(test_mathcv test.cpp ${TESTS_FILES} )

    link_directories(${CMAKE_CURRENT_BINARY_DIR}/UnitTest++)
    target_link_libraries(test_mathcv PRIVATE ${PROJECT_NAME} UnitTest++)

    add_test(
        NAME ${PROJECT_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND test_mathcv
        )
    set_tests_properties(${PROJECT_NAME} PROPERTIES TIMEOUT 60)
	
        if (${PROJECT_NAME}_RUN_TESTS)
                add_custom_command(TARGET test_mathcv
                        POST_BUILD COMMAND test_mathcv
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Running test_mathcv")
        endif()

endif()

