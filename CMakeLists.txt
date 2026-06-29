cmake_minimum_required(VERSION 3.16)
project(lacam2-project CXX)

option(LACAM2_BUILD_TESTS "Build test targets" OFF)

add_subdirectory(./lacam2)
add_subdirectory(./third_party/argparse)
if(LACAM2_BUILD_TESTS)
  add_subdirectory(./third_party/googletest)
endif()

# SFML for GUI visualizer
find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED)

add_executable(main
    main.cpp
    gui.cpp
)
target_compile_features(main PUBLIC cxx_std_17)
target_include_directories(main PRIVATE
    /home/galko/.local/include
)
target_link_libraries(main
    lacam2
    argparse
    sfml-graphics sfml-window sfml-system
    /home/galko/.local/lib/libCBSH2.so
)

# test
if(LACAM2_BUILD_TESTS)
  set(TEST_MAIN_FUNC ./third_party/googletest/googletest/src/gtest_main.cc)
  file(GLOB TEST_FILES "./tests/test_*.cpp")
  file(GLOB DG_TEST_FILES "./tests/*_test.cpp")
  list(APPEND TEST_FILES ${DG_TEST_FILES})
  foreach(file ${TEST_FILES})
    string(REGEX MATCH "test\_[^\.]+" name "${file}")
    if(NOT name)
      string(REGEX MATCH "[^/]+_test" name "${file}")
    endif()
    add_executable(${name} ${file} ${TEST_MAIN_FUNC})
    target_link_libraries(${name} lacam2 gtest)
  endforeach()
  add_executable(test_all ${TEST_MAIN_FUNC} ${TEST_FILES})
  target_link_libraries(test_all lacam2 gtest)
endif()
