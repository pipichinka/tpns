#include "NumCpp.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include "perceptron.hpp"
#include "csv.hpp"



std::vector<std::string> parse_in_comma_sep(char* in){
    std::vector<std::string> result;
    char* current_line_start = in;
    char* line_end = in + strlen(in);
    while (current_line_start < line_end){
        char* sep_pos = strchrnul(current_line_start, ',');
        std::string value(current_line_start, 0, sep_pos - current_line_start);
        result.push_back(value);
        current_line_start = sep_pos + 1;
    }
    return result;
}


std::vector<int> transform_s_to_i_vec(const std::vector<std::string>& in){
    std::vector<int> res;
    for (const std::string& s: in){
        res.push_back(std::stoi(s));
    }
    return res;
}


std::vector<layer_interface*> make_layers(const std::vector<int> in_layers, int train_layer_size, int target_layer_size){
    std::vector<layer_interface*> res;
    if (in_layers.size() == 0){
        res.emplace_back(new layer<activation_relu>(train_layer_size, target_layer_size));
        return res;
    }
    res.emplace_back(new layer<activation_relu>(train_layer_size, in_layers[0]));
    for (int i = 0; i < in_layers.size() - 1; i++){
        res.emplace_back(new layer<activation_relu>(in_layers[i], in_layers[i + 1]));
    }
    res.emplace_back(new layer<activation_x>(in_layers.back(), target_layer_size));
    return res;
}


void test_perspetron(perceptron& p, std::vector<nc::NdArray<double>>& in_data, std::vector<nc::NdArray<double>>& out_data){
    double error_sum = 0;
    for (size_t i = in_data.size() * 0.8; i < in_data.size(); i++){
        nc::NdArray<double> diff = p.solve(in_data[i]) - out_data[i];
        error_sum += nc::dot(diff, diff.transpose())(0,0);
    }
    std::cout << " test MSE " << std::sqrt(error_sum ) / ((double) in_data.size() * 0.2) << std::endl;

    nc::NdArray<double> avarage(out_data[in_data.size() * 0.8]);
    for (size_t i = in_data.size() * 0.8 + 1; i < out_data.size(); i++){
        avarage += out_data[i];
    }
    avarage /= (double) out_data.size() * 0.2;

    double diff_from_avarage = 0;
    for (size_t i = 0; i < out_data.size(); i++){
        nc::NdArray<double> diff = out_data[i] - avarage;
        diff_from_avarage += nc::dot(diff, diff.transpose())(0,0);
    }

    std::cout << " test r2 score " << 1 - error_sum / diff_from_avarage << std::endl;

    error_sum = 0;
    for (size_t i = 0; i < in_data.size(); i++){
        nc::NdArray<double> diff = p.solve(in_data[i]) - out_data[i];
        error_sum += nc::dot(diff, diff.transpose())(0,0);
    }
    std::cout << " all MSE " << std::sqrt(error_sum) / (double) in_data.size() << std::endl;

    avarage = out_data[0];
    for (size_t i = 1; i < out_data.size(); i++){
        avarage += out_data[i];
    }
    avarage /= (double) out_data.size();

    diff_from_avarage = 0;
    for (size_t i = 0; i < out_data.size(); i++){
        nc::NdArray<double> diff = out_data[i] - avarage;
        diff_from_avarage += nc::dot(diff, diff.transpose())(0,0);
    }

    std::cout << " all r2 score " << 1 - error_sum / diff_from_avarage << std::endl;
}


// file train(comma sep list) target(comma sep list) in_layers(comma sep list) epoch
int main(int argc, char** argv){
    if (argc != 6){
        std::cerr << "usage file train(comma sep list) target(comma sep list) epoch\n";
        return 1;
    }

    csv* file = new csv(argv[1]);

    std::vector<std::string> train_columns(parse_in_comma_sep(argv[2]));
    std::vector<std::string> target_columns(parse_in_comma_sep(argv[3]));

    std::vector<nc::NdArray<double>> train_data(file->get_data_by_names(train_columns));
    std::vector<nc::NdArray<double>> target_data(file->get_data_by_names(target_columns));

    std::vector<layer_interface*> layers(make_layers(
        transform_s_to_i_vec(parse_in_comma_sep(argv[4])), 
        train_columns.size(), 
        target_columns.size()));

    error_counter_interface* error_counter = new common_error_counter();
    perceptron p(layers, error_counter);
    p.train(train_data, target_data, train_data.size() * 0.8 , std::stoi(argv[5]), 0.005);

    test_perspetron(p, train_data, target_data);

    return 0;
}