#include "layer_interface.hpp"


class flatten_layer: public layer_interface{
    int in_size;
    int in_matrix_size;
public:
    virtual std::vector<nc::NdArray<double>> forward(const std::vector<nc::NdArray<double>>& vec){
        in_size = vec.size();
        in_matrix_size = vec[0].numCols();
        std::vector<nc::NdArray<double>> res;
        nc::NdArray<double> in_res(1, in_size * in_matrix_size * in_matrix_size);
        for (int i = 0; i < in_size; i++){
            for (int y = 0; y < in_matrix_size; y++){
                for (int x = 0; x < in_matrix_size; x++){
                    int index = i * (in_matrix_size * in_matrix_size) + y * in_matrix_size + x;
                    in_res(0, index) = vec[i](y, x);
                }
            }
        }
        res.emplace_back(in_res);
        return res;
    }

    virtual std::vector<nc::NdArray<double>> backward(const std::vector<nc::NdArray<double>>& error, double learn_rate){
        std::vector<nc::NdArray<double>> res;
        for (int i = 0; i < in_size; i++){
            nc::NdArray<double> in_res(in_matrix_size, in_matrix_size);
            for (int y = 0; y < in_matrix_size; y++){
                for (int x = 0; x < in_matrix_size; x++){
                    int index = i * (in_matrix_size * in_matrix_size) + y * in_matrix_size + x;
                    in_res(y, x) = error[0](0, index);
                }
            }
            res.emplace_back(in_res);
        }
        return res;
    }
};