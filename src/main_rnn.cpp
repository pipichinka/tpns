#include <NumCpp.hpp>
#include "csv.hpp"
#include "lstm_layer.hpp"
#include "rnn_layer.hpp"
#include "gru_layer.hpp"
#include "perceptron.hpp"

struct data{
    std::vector<nc::NdArray<double>> train;
    std::vector<nc::NdArray<double>> target;

    data(std::vector<nc::NdArray<double>>&& train, std::vector<nc::NdArray<double>>&& target): train(train), target(target){}

    void shuffle(){
        nc::NdArray<int> indexes = nc::arange<int>(train.size());
        nc::random::seed(42);
        nc::random::shuffle(indexes);
        std::vector<nc::NdArray<double>> train_shuffle;
        std::vector<nc::NdArray<double>> target_shuffle;
        for (int index: indexes){
            train_shuffle.emplace_back(std::move(train[index]));
            target_shuffle.emplace_back(std::move(target[index]));
        }
        train = train_shuffle;
        target = target_shuffle;
    }
};

data transform_data(const std::vector<nc::NdArray<double>>& train,const std::vector<nc::NdArray<double>>& target, int group_size){
    std::vector<nc::NdArray<double>> res_train;
    std::vector<nc::NdArray<double>> res_target;
    for (int i = 0; i < train.size() - group_size; i++){
        nc::NdArray<double> cur_train(group_size, train[0].numCols());
        for (int j = 0; j < group_size; j++){
            cur_train.put(j, nc::Slice(0, cur_train.numCols()), train[j + i]);
        }
        res_train.push_back(cur_train);
        res_target.push_back(target[i + group_size]);

    }
    return data(std::move(res_train), std::move(res_target));
}


nc::NdArray<int> make_cat(const nc::NdArray<double>& in, double threshold){
    nc::NdArray<int> res(in.shape());
    double max = in.max()(0,0);
    for (int i = 0; i < in.shape().cols; i++){
        if (in(0, i) == max){
            res(0, i) = 1;
        } else {
            res(0, i) = 0;
        }
    }
    return res;
}


void test_perspetron(perceptron& p, std::vector<nc::NdArray<double>>& in_data, std::vector<nc::NdArray<double>>& out_data, const std::vector<std::string>& out_data_columns){

    nc::NdArray<int> tp = nc::zeros<int>(out_data[0].shape());
    nc::NdArray<int> tn = tp;
    nc::NdArray<int> fp = tp;
    nc::NdArray<int> fn = tp;
    for (size_t i = in_data.size() * 0.9; i < in_data.size(); i++){
        auto res =make_cat(p.solve(in_data[i]), 0.5);
        // std::cout << "pred" << res;
        // std::cout << "true" << out_data[i];
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

    std:: cout << tp << tn << fp << fn;
    for (int j = 0; j < out_data[0].shape().cols; j++){
        
        double precision = ((double) tp(0, j)) / ((double) tp(0, j) + fp(0, j));
        std::cout << "precision of " << out_data_columns[j] << ": " <<  precision << std::endl;

        double recall = ((double) tp(0, j)) / ((double) tp(0, j) + fn(0, j));
        std::cout << "recall of " << out_data_columns[j] << ": " << recall << std::endl;

        double f_score = 2.0 * (precision * recall) / (precision + recall);
        std::cout << "f score " << out_data_columns[j] << ": " << f_score << std::endl;
    }

    std::cout << std::endl << std::endl;
}    

int main(int argc, char** argv){

    csv* csv_train = new csv("../lab3_train.csv");
    csv* csv_target = new csv("../lab3_target.csv");

    data df = transform_data(csv_train->get_data(), csv_target->get_data(), 6);
    df.shuffle();
    std::vector<layer_interface*> layers = {
        new rnn_layer(df.train[0].numCols(), 24, 24, 0.3),
        new relu_layer(),
        new rnn_layer(24, 12, 10, 0.3),
        new relu_layer(),
        new rnn_out_layer<activation_softmax>(10, 3, 0.3)
    };
    perceptron p(layers, new common_error_counter());
    p.train(df.train, df.target, df.target.size() * 0.9, std::stoi(argv[2]), std::stod(argv[1]));
    test_perspetron(p, df.train, df.target, csv_target->get_head());
    
}