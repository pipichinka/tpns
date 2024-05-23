cmake_minimum_required(VERSION 3.20)

project("tpns" CXX)
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall")
SET(CMAKE_CXX_FLAGS  "-O2")
add_executable(lab2_laptops src/main_laptops.cpp src/layer.hpp src/perceptron.hpp src/perceptron.cpp src/csv.hpp)

add_executable(lab2_mashrooms src/main_mashrooms.cpp src/layer.hpp src/perceptron.hpp src/perceptron.cpp src/csv.hpp)

add_executable(lab3_rnn src/main_rnn.cpp src/csv.hpp src/layer.hpp src/perceptron.hpp src/rnn_layer.hpp src/perceptron.cpp src/lstm_layer.hpp
                src/gru_layer.hpp)

add_executable(lab4 src/main_lab4.cpp src/padding_layer.hpp src/layer_interface.hpp src/convolution_layer.hpp src/avarage_pooling_layer.hpp 
                    src/flatten_layer.hpp src/dense_layer.hpp src/model.cpp src/model.hpp src/max_pooling_layer.hpp)

add_executable(test src/test.cpp)
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

target_link_libraries(lab3_rnn NumCpp::NumCpp)

target_link_libraries(lab4 NumCpp::NumCpp)

target_link_libraries(test NumCpp::NumCpp)