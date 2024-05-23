#include "layer_interface.hpp"


class max_pooling_layer: public layer_interface{

    int in_size;
    int in_matrix_size;
    int pooling_size;
public:

    max_pooling_layer(int in_size, int in_matrix_size, int pooling_size):
    in_size(in_size), in_matrix_size(in_matrix_size), pooling_size(pooling_size)
    {

    }

    virtual std::vector<nc::NdArray<double>> forward(const std::vector<nc::NdArray<double>>& in_vec){
        std::vector<nc::NdArray<double>> res;
        for (int i = 0; i < in_vec.size(); i++){
            nc::NdArray<double> cur_v(in_matrix_size / pooling_size, in_matrix_size / pooling_size);
            for (int y = 0; y < in_matrix_size; y += pooling_size){
                for (int x = 0; x < in_matrix_size; x+= pooling_size){
                    double max = -INFINITY;
                     
                    for (int in_y = 0; in_y < pooling_size; in_y++){
                        for (int in_x = 0; in_x < pooling_size; in_x++){
                            if (max < in_vec[i](y + in_y, x + in_x)){
                                max = in_vec[i](y + in_y, x + in_x);
                            }
                        }
                    }
                    cur_v(y / pooling_size, x / pooling_size) = max;

                }
            }
            res.emplace_back(cur_v);
        }
        return res;
    }

    virtual std::vector<nc::NdArray<double>> backward(const std::vector<nc::NdArray<double>>& error, double learn_rate){
        std::vector<nc::NdArray<double>> res;
        double mul = (double) pooling_size * pooling_size;
        for (int i = 0; i < error.size(); i++){
            nc::NdArray<double> cur_v(in_matrix_size, in_matrix_size);
            for (int y = 0; y < in_matrix_size; y++){
                int in_y = y / pooling_size;
                for (int x = 0; x < in_matrix_size; x++){
                    cur_v(y, x) = error[i](in_y, x / pooling_size) / mul;
                }
            }
            res.emplace_back(cur_v);
        }   
        return res; 
    }
};