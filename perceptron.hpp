#include "layer.hpp"
#include <vector>


class error_counter_interface{
public:
    virtual nc::NdArray<double> count(const nc::NdArray<double>& a, const nc::NdArray<double>& b) = 0;
};


class common_error_counter: public error_counter_interface{
    virtual nc::NdArray<double> count(const nc::NdArray<double>& a, const nc::NdArray<double>& b){
        return a - b;
    }
};


class evclid_error_counter: public error_counter_interface{
    virtual nc::NdArray<double> count(const nc::NdArray<double>& a, const nc::NdArray<double>& b){
        return nc::sqrt((a-b) * (a-b));
    }
};


class category_cross_entropy_error_counter: public error_counter_interface{
    virtual nc::NdArray<double> count(const nc::NdArray<double>& a, const nc::NdArray<double>& b){
        auto a_clip = a.clip(1e-15, 1 - 1e-15);
        return (a_clip - b) / (double) a.shape().cols;
    }
};


class perceptron{
    std::vector<layer_interface*> layers;
    error_counter_interface* error_counter;
public:
    perceptron(std::vector<layer_interface*>& layers, error_counter_interface* error_counter): layers(layers), error_counter(error_counter){}

    void train(const std::vector<nc::NdArray<double>>& in_data, const std::vector<nc::NdArray<double>>& out_data,size_t data_size, int epoch, double learn_rate);

    nc::NdArray<double> solve(const nc::NdArray<double>& in);
};