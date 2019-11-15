project(eigen_quickref)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

include_directories(eigen-337)

add_executable(quickref_100_numpy quickref_100_numpy.cpp)