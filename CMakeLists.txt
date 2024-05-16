cmake_minimum_required(VERSION 3.20)

project("tpns" CXX)
add_executable(lab2_laptops main_laptops.cpp layer.hpp perceptron.hpp perceptron.cpp csv.hpp)

add_executable(lab2_mashrooms main_mashrooms.cpp layer.hpp perceptron.hpp perceptron.cpp csv.hpp)
include(FetchContent)
FetchContent_Declare(NumCpp
        GIT_REPOSITORY https://github.com/dpilger26/NumCpp
        GIT_TAG Version_2.12.1)
FetchContent_MakeAvailable(NumCpp)

target_link_libraries(lab2_laptops
    NumCpp::NumCpp
)

target_link_libraries(lab2_mashrooms PUBLIC
    NumCpp::NumCpp
)