cmake_minimum_required(VERSION 3.20)
project(cfr_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# The engine itself is plain C++; only bindings.cpp knows about Python.
add_library(cfr_core STATIC cards.cpp cfr.cpp)
set_target_properties(cfr_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# No fused multiply-adds, so element-wise steps round exactly as numpy's do (tests/test_cpp.py).
target_compile_options(cfr_core PRIVATE -Wall -Wextra -ffp-contract=off)

pybind11_add_module(cfr_cpp bindings.cpp)
target_link_libraries(cfr_cpp PRIVATE cfr_core)
