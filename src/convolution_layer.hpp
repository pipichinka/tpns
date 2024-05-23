#include "layer_interface.hpp"

class convolution_layer:public layer_interface{
    nc::NdArray<double>* kernel_matrix;
    int in_size;
    int in_matrix_size;
    int out_size;
    int kernel_size;
    std::vector<nc::NdArray<double>> b;
    std::vector<nc::NdArray<double>> input_data;

    nc::NdArray<double> applyCorrValid(const nc::NdArray<double>& in, const nc::NdArray<double>& kernel){
        nc::NdArray<double> res(nc::Shape(in.numCols() - kernel.numCols() + 1, in.numCols() - kernel.numCols() + 1));
        if (kernel.numCols() % 2 == 0){
            int border_step = (kernel.numCols()) / 2;
            for (int i = border_step; i <= in.numRows() - border_step; i++){
                for (int j = border_step; j <= in.numCols() - border_step; j++){
                    double sum = 0.0;
                    for (int in_x = - border_step; in_x < border_step; in_x++){
                        for (int in_y = -border_step; in_y < border_step; in_y++){
                            int in_y_index = in_y + border_step;
                            int in_x_index = in_x + border_step;
                            sum += in(i + in_y, j + in_x) * kernel(in_y_index, in_x_index);
                        }
                    }
                    res(i - border_step, j - border_step) = sum;    
                }
            }
            return res;
        }
        int border_step = (kernel.numCols() - 1) / 2;
        for (int i = border_step; i < in.numRows() - border_step; i++){
            for (int j = border_step; j < in.numCols() - border_step; j++){
                double sum = 0.0;
                for (int in_x = - border_step; in_x <= border_step; in_x++){
                    for (int in_y = -border_step; in_y <= border_step; in_y++){
                        int in_y_index = in_y + border_step;
                        int in_x_index = in_x + border_step;
                        sum += in(i + in_y, j + in_x) * kernel(in_y_index, in_x_index);
                    }
                }
                res(i - border_step, j - border_step) = sum;    
            }
        }
        return res;
    }


    nc::NdArray<double> applyCorrFull(const nc::NdArray<double>& in, const nc::NdArray<double>& kernel){
        int border_step = (kernel.numCols() - 1);
        nc::NdArray<double> res(nc::Shape(in.numRows() + kernel.numRows() - 1, in.numCols() + kernel.numCols() - 1));
        for (int i = 0; i < res.numRows(); i++){
            for (int j = 0; j < res.numCols(); j++){
                double sum = 0.0;
                for (int in_x = - border_step; in_x < 1; in_x++){
                    for (int in_y = -border_step; in_y < 1; in_y++){
                        int in_y_index = in_y + border_step;
                        int in_x_index = in_x + border_step;
                        int matrix_indexi = i + in_y;
                        int matrix_indexj = j + in_x;
                        if ( matrix_indexi < 0 || matrix_indexi >= in.numRows() || matrix_indexj < 0 || matrix_indexj >= in.numCols()){
                            continue;
                        }
                        sum += in(matrix_indexi, matrix_indexj) * kernel(in_y_index, in_x_index);
                    }
                }
                res(i, j) = sum;    
            }
        }
        
        return res;
    }


    nc::NdArray<double> turnMatrix(const nc::NdArray<double>& matrix){
        nc::NdArray<double> res(matrix.shape());
        for (int i = 0; i < matrix.numRows(); i++){
            for (int j = 0; j < matrix.numCols(); j++){
                res(matrix.numRows() - i - 1, j) = matrix(i, j);
            }
        }
        return res;
    }


public:
    convolution_layer(int in_size, int in_matrix_size, int out_size, int kernel_size):
    in_size(in_size),
    in_matrix_size(in_matrix_size),
    out_size(out_size),
    kernel_size(kernel_size)
    {
        kernel_matrix = new nc::NdArray<double>[in_size * out_size];
        for (int i = 0; i < in_size * out_size; i++){
            kernel_matrix[i] = nc::random::randN<double>(nc::Shape(kernel_size, kernel_size)) * 0.5;
        }

        for (int i = 0; i < out_size; i++){
            b.emplace_back(nc::random::randN<double>(nc::Shape(in_matrix_size - kernel_size + 1, in_matrix_size - kernel_size + 1)) * 0.5);
        }
    }


    virtual std::vector<nc::NdArray<double>> forward(const std::vector<nc::NdArray<double>>& x){
        input_data = x;
        std::vector<nc::NdArray<double>> res(b);
        for (int i = 0; i < out_size; i++){
            for (int j = 0; j < in_size; j++){
                res[i] += applyCorrValid(x[j], kernel_matrix[i * in_size + j]);
            }
        }
        return res;
    }

    virtual std::vector<nc::NdArray<double>> backward(const std::vector<nc::NdArray<double>>& error, double learn_rate){
        std::vector<nc::NdArray<double>> d_in;
        for (int i = 0; i < in_size; i++){
            d_in.emplace_back(nc::zeros<double>(in_matrix_size, in_matrix_size));
        }

        for (int i = 0; i < out_size; i++){
            for (int j = 0; j < in_size; j++){
                d_in[j] += applyCorrFull(error[i], turnMatrix(kernel_matrix[i * in_size + j]));
                kernel_matrix[i * in_size + j] -= learn_rate * applyCorrValid(input_data[j], error[i]);
            }
            b[i] -= learn_rate * error[i]; 
        }

        return d_in;
    }
};

class sigmoid_layer: public layer_interface{
    std::vector<nc::NdArray<double>> save_pred;
public:
    virtual std::vector<nc::NdArray<double>> forward(const std::vector<nc::NdArray<double>>& x){
        save_pred.clear();
        for (int i = 0; i < x.size(); i++){
            save_pred.emplace_back(1.0 / (1.0 - nc::exp(-x[i])));
        }
        return save_pred;
    }

    virtual std::vector<nc::NdArray<double>> backward(const std::vector<nc::NdArray<double>>& error, double learn_rate){
        for (int i = 0; i < error.size(); i++){
            save_pred[i] = error[i] * save_pred[i] * (1.0 - save_pred[i]);
        }
        return save_pred;
    }
};


class relu_layer: public layer_interface{
    std::vector<nc::NdArray<double>> save_pred;
public:
    virtual std::vector<nc::NdArray<double>> forward(const std::vector<nc::NdArray<double>>& x){
        save_pred.clear();
        for (int i = 0; i < x.size(); i++){
            save_pred.emplace_back(x[i]);
            nc::applyFunction(save_pred[i], std::function([](double v) -> double{
                return std::max(0.0, v);
            }));
        }
        return save_pred;
    }

    virtual std::vector<nc::NdArray<double>> backward(const std::vector<nc::NdArray<double>>& error, double learn_rate){
        for (int i = 0; i < error.size(); i++){
            nc::applyFunction(save_pred[i], std::function([](double v) -> double{
                return v >= 0.0 ? 1.0:0.0;
            }));
            save_pred[i] = error[i] * save_pred[i];
        }
        return save_pred;
    }
};