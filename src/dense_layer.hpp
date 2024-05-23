#include "layer_interface.hpp"

template<typename activation>
class dense_layer: public layer_interface{
    nc::NdArray<double> x; // 1Xin
    nc::NdArray<double> W; // inXout
    nc::NdArray<double> b; // 1Xout
    nc::NdArray<double> t; // 1Xout
public:     
    dense_layer(int in_layer, int out_layer): 
    x(nc::zeros<double>(1, in_layer)), 
    W(nc::random::randN<double>(nc::Shape(in_layer, out_layer)) * 0.5), 
    b(nc::zeros<double>(1, out_layer)){

    }
    virtual std::vector<nc::NdArray<double>> forward(const std::vector<nc::NdArray<double>>& x){
        this->x = x[0];
        nc::NdArray<double> res = nc::dot(x[0], W) + b; // (1Xin * inXout) + 1Xout
        t = res; // 1Xout
        res = activation::forward(res);
        return {res};
    }

    virtual std::vector<nc::NdArray<double>> backward(const std::vector<nc::NdArray<double>>& error, double learn_rate){
        t = activation::derivative(t);
        nc::NdArray<double> dt = error[0] * t; // 1xout
        nc::NdArray<double> dx = nc::dot(dt, W.transpose()); //1Xout * outXin
        b -= learn_rate * dt;
        W -= learn_rate * nc::dot(x.transpose(), dt); //inX1 * 1Xout
        return {dx};
    }
};