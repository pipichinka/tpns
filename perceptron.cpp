#include "perceptron.hpp"

void perceptron::train(const std::vector<nc::NdArray<double>>& in_data, const std::vector<nc::NdArray<double>>& out_data, size_t data_size, int epoch, double learn_rate){
    for (int i = 0; i < epoch; i++){
        double avarage_error = 0;
        for (size_t j = 0; j < data_size; j++){
            
            nc::NdArray<double> learn_vector = in_data[j];
            //std::cout << "learn init " << learn_vector;
            for (layer_interface* l: layers){
                learn_vector = l->forward(learn_vector);
                //std::cout << "layer forward" << learn_vector;
            }
            learn_vector = error_counter->count(learn_vector, out_data[j]);
            //std::cout << "error " << learn_vector;
            double error = 0;
            for (double error_i: learn_vector){
                error += std::abs(error_i);
            }
            avarage_error += error / (double) data_size; 
            for (auto it = layers.rbegin(); it != layers.rend(); ++it){
                learn_vector = (*it)->backward(learn_vector, learn_rate);
                //std::cout << "bacward " << learn_vector;
            }

        }

        std::cout << "average error on " << i << " epoch: " << avarage_error << std::endl;
    }
}


nc::NdArray<double> perceptron::solve(const nc::NdArray<double>& in){
    nc::NdArray<double> learn_vector = in;
    for (layer_interface* l: layers){
        learn_vector = l->forward(learn_vector);
    }
    return learn_vector;
}