#include <NumCpp.hpp>
#include "layer.hpp"
#ifndef RNN_LAYER_H
#define RNN_LAYER_H


class rnn_layer: public layer_interface{
    nc::NdArray<double> save_x;
    nc::NdArray<double> save_h;
    nc::NdArray<double> save_Y;
    nc::NdArray<double> W_h;
    nc::NdArray<double> U;
    nc::NdArray<double> b_h;
    nc::NdArray<double> W_y;
    nc::NdArray<double> b_y;
public:

    rnn_layer(int in_size, int hidden_size, int out_size, double ws):
    W_h(nc::random::uniform<double>(nc::Shape(in_size, hidden_size), -ws, ws)),
    U(nc::random::uniform<double>(nc::Shape(hidden_size, hidden_size), -ws, ws)),
    b_h(nc::zeros<double>(1, hidden_size)),
    W_y(nc::random::uniform<double>(nc::Shape(hidden_size, out_size), -ws, ws)),
    b_y(nc::zeros<double>(1, out_size))
    {
    }
    virtual nc::NdArray<double> forward(const nc::NdArray<double>& x){
        uint32_t rows = x.shape().rows;
        uint32_t hidden_cols = U.shape().cols;
        save_x = x;
        save_h = nc::NdArray<double>(rows, U.shape().cols);
        save_Y = nc::NdArray<double>(rows, b_y.numCols());
        nc::NdArray<double> prev_h = nc::zeros<double>(1, U.shape().cols);
        for (uint32_t i = 0; i < rows; i++){
            nc::NdArray<double> x_i = nc::dot(x.row(i), W_h); //1Xin * inXhi
            prev_h = x_i + nc::dot(prev_h, U) + b_h; // 1Xhi
            save_h.put(i, nc::Slice(0, hidden_cols), prev_h);
            prev_h = activation_tanh::forward(prev_h);
            nc::NdArray<double> cur_Y = nc::dot(prev_h, W_y) + b_y;
            save_Y.put(i, nc::Slice(0, save_Y.numCols()), cur_Y);
        }

        return save_Y;
    }

    virtual nc::NdArray<double> backward(const nc::NdArray<double> error, double learn_rate){
        nc::NdArray<double> d_b_y = nc::zeros<double>(b_y.shape());
        nc::NdArray<double> d_W_y = nc::zeros<double>(W_y.shape());
        nc::NdArray<double> d_b_h = nc::zeros<double>(b_h.shape());
        nc::NdArray<double> d_U = nc::zeros<double>(U.shape());
        nc::NdArray<double> d_W_h = nc::zeros<double>(W_h.shape());

        nc::NdArray<double> d_hidden_error = nc::zeros<double>(1, U.numCols());
        nc::NdArray<double> d_x = nc::zeros<double>(save_x.shape());

        for (int i = save_h.numRows() - 1; i >= 0; --i){
            nc::NdArray<double> h_i = save_h.row(i);
            nc::NdArray<double> t_i = error.row(i); // 1Xout
            d_b_y += t_i;
            d_W_y += nc::dot(h_i.transpose(), t_i); // hiX1 * 1Xout
            //                             1Xhi                         1Xout * outXhi          1Xhi
            nc::NdArray<double> d_h_i = activation_tanh::derivative(h_i) * (nc::dot(t_i, W_y.transpose()) + d_hidden_error);
            if (i > 0){
                //                    hiX1                  * 1Xhi 
                d_U += nc::dot(save_h.row(i - 1).transpose(), d_h_i);
            }
            d_b_h += d_h_i;
            //                inX1                   * 1Xhi
            d_W_h += nc::dot(save_x.row(i).transpose(), d_h_i);
            //                      1Xhi   * hiXhi
            d_hidden_error = nc::dot(d_h_i, d_U.transpose());
            //                                              1Xhi * hiXin
            d_x.put(i, nc::Slice(0, d_x.numCols()), nc::dot(d_h_i, W_h.transpose()));
        }

        b_y -= learn_rate * d_b_y; 
        b_h -= learn_rate * d_b_h; 
        W_h -= learn_rate * d_W_h; 
        W_y -= learn_rate * d_W_y; 
        U -= learn_rate * d_U; 

        return d_x;
    }
};


template<typename activation>
class rnn_out_layer: public layer_interface{
    nc::NdArray<double> x; // 1Xin
    nc::NdArray<double> W; // inXout
    nc::NdArray<double> b; // 1Xout
    nc::NdArray<double> t; // 1Xout
public:     
    rnn_out_layer(int in_layer, int out_layer, double ws):
    W(nc::random::uniform<double>(nc::Shape(in_layer, out_layer), -ws, ws)), 
    b(nc::zeros<double>(1, out_layer)){

    }
    //x 1Xin
    virtual nc::NdArray<double> forward(const nc::NdArray<double>& x){
        this->x = x;
        nc::NdArray<double> res = nc::dot(x.row(x.numRows() - 1), W) + b; // (1Xin * inXout) + 1Xout
        t = res; // 1Xout
        res = activation::forward(res);
        return res;
    }

    
    //error 1xout
    virtual nc::NdArray<double> backward(const nc::NdArray<double> error, double learn_rate){

        t = activation::derivative(t);
        nc::NdArray<double> dx = nc::zeros<double>(x.shape());
        nc::NdArray<double> dt = error * t; // 1xout
        dx.put(x.numRows() - 1, nc::Slice(0, x.numCols()), nc::dot(dt, W.transpose())); //1Xout * outXin
        b -= learn_rate * dt;
        W -= learn_rate * nc::dot(x.row(x.numRows() - 1).transpose(), dt); //inX1 * 1Xout
        return dx;
    }
};


#endif