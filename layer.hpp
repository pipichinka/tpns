#include "NumCpp.hpp"
#include <algorithm>
//relu

class actrivation_relu{
public:
    double inline operator()(double x){
        return std::max(0.0, x);
    }

    double inline derivative(double x){
        return x >= 0.0 ? 1.0:0.0;
    }
};

class activation_x{
public:
    double inline operator()(double x){
        return x;
    }

    double inline derivative(double x){
        return 1;
    }
};


class layer_interface{
public:
    virtual nc::NdArray<double> forward(const nc::NdArray<double>& x) = 0;

    virtual nc::NdArray<double> backward(const nc::NdArray<double> error, double learn_rate) = 0;
};


template<typename activation>
class layer: public layer_interface{
    nc::NdArray<double> x; // 1Xin
    nc::NdArray<double> W; // inXout
    nc::NdArray<double> b; // 1Xout
    nc::NdArray<double> t; // 1Xout
public:     
    layer(int in_layer, int out_layer): 
    x(nc::zeros<double>(1, in_layer)), 
    W(nc::random::rand<double>(nc::Shape(in_layer, out_layer))), 
    b(nc::random::rand<double>(nc::Shape(1, out_layer))){

    }
    //x 1Xin
    virtual nc::NdArray<double> forward(const nc::NdArray<double>& x){
        activation a;
        this->x = x;
        nc::NdArray<double> res = nc::dot(x, W) + b; // (1Xin * inXout) + 1Xout
        t = res; // 1Xout
        for (auto it = res.begin(); it != res.end(); ++it){
            *(it) = a(*(it));
        }
        return res;
    }

    
    //error 1xout
    virtual nc::NdArray<double> backward(const nc::NdArray<double> error, double learn_rate){
        activation a;
        for (auto it = t.begin(); it != t.end(); ++it){
            *it = a.derivative(*it);
        }
        nc::NdArray<double> dt = error * t; // 1xout
        nc::NdArray<double> dx = nc::dot(dt, W.transpose()); //1Xout * outXin
        b -= learn_rate * dt;
        W -= learn_rate * nc::dot(x.transpose(), dt); //inX1 * 1Xout
        return dx;
    }
};