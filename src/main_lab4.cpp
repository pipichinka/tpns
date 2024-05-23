#include <NumCpp.hpp>
#include <string>
#include "csv.hpp"

#include "padding_layer.hpp"
#include "convolution_layer.hpp"
#include "avarage_pooling_layer.hpp"
#include "flatten_layer.hpp"
#include "dense_layer.hpp"

#include "model.hpp"

nc::NdArray<int> make_cat(const nc::NdArray<double>& in, double threshold){
    nc::NdArray<int> res(in.shape());
    for (int i = 0; i < in.shape().cols; i++){
        if (in(0, i) < threshold){
            res(0, i) = 0;
        } else {
            res(0, i) = 1;
        }
    }
    return res;
}

void test_model(model& p, const std::vector<nc::NdArray<double>>& in_data, const std::vector<nc::NdArray<double>>& out_data, const std::vector<std::string>& out_data_columns){

    nc::NdArray<int> tp = nc::zeros<int>(out_data[0].shape());
    nc::NdArray<int> tn = tp;
    nc::NdArray<int> fp = tp;
    nc::NdArray<int> fn = tp;
    for (size_t i = 0; i < in_data.size(); i++){
        auto res =make_cat(p.solve(in_data[i]), 0.3);
        std::cout << "pred" << res;
        std::cout << "true" << out_data[i];
        for (int j = 0; j < out_data[0].shape().cols; j++){
            if (res(0, j) == 1 && out_data[i](0, j) == 1){
                tp(0, j) += 1;
            }
            if (res(0, j) == 0 && out_data[i](0, j) == 0){
                tn(0, j) += 1;
            }
            if (res(0, j) == 1 && out_data[i](0, j) == 0){
                fp(0, j) += 1;
            }
            if (res(0, j) == 0 && out_data[i](0, j) == 1){
                fn(0, j) += 1;
            }
        }
    }

    for (int j = 0; j < out_data[0].shape().cols; j++){        
        double precision = ((double) tp(0, j)) / ((double) tp(0, j) + fp(0, j));
        std::cout << "precision of " << out_data_columns[j] << ": " <<  precision << std::endl;

        double recall = ((double) tp(0, j)) / ((double) tp(0, j) + fn(0, j));
        std::cout << "recall of " << out_data_columns[j] << ": " << recall << std::endl;

        double f_score = 2.0 * (precision * recall) / (precision + recall);
        std::cout << "f score " << out_data_columns[j] << ": " << f_score << std::endl;
        std::cout << std::endl;
    }

    std::cout << std::endl << std::endl;
}  

int main(int argc, char** argv){
    csv train_in("../lab4_train_in.csv");
    csv train_out("../lab4_train_out.csv");
    std::cout << "dataset loaded" << std::endl;
    std::vector<layer_interface*> layers ({
        new padding_layer(2),
        new convolution_layer(1, 32, 6, 5),
        new avarage_pooling_layer(6, 28, 2),
        new convolution_layer(6, 14, 16, 5),
        new avarage_pooling_layer(16, 10, 2),
        new flatten_layer(),
    });
    layers.emplace_back(new dense_layer<activation_x>(400, 120));
    // layers.emplace_back(new sigmoid_layer());
    layers.emplace_back(new dense_layer<activation_x>(120, 84));
    // layers.emplace_back(new sigmoid_layer());
    layers.emplace_back(new dense_layer<activation_softmax>(84, 10));
    model model(layers, new category_cross_entropy_error_counter());
    std::cout << "starting train" << std::endl;
    model.train(train_in.get_data(), train_out.get_data(),  train_out.get_data().size(), std::stoi(argv[1]), std::stoi(argv[2]));
    csv test_in("../lab4_test_in.csv");
    csv test_out("../lab4_test_out.csv");
    std::cout << "test data size " << test_in.get_data().size() << std::endl;
    test_model(model, test_in.get_data(), test_out.get_data(), test_out.get_head());
    return 0;
}

