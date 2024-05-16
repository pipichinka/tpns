#include "NumCpp.hpp"
#include <algorithm>

class actrivation_relu{
public:
    static nc::NdArray<double> inline forward(const nc::NdArray<double>& x){
        nc::NdArray<double> res = x;
        nc::applyFunction(res, std::function([](double v) -> double {return std::max(0.0, v);}));
        return res;
    }

    static nc::NdArray<double> inline derivative(const nc::NdArray<double>& x){
        nc::NdArray<double> res = x;
        nc::applyFunction(res, std::function([](double v) -> double {return v >= 0 ? 1.0:0.0;}));
        return res;
    }
};

class activation_x{
public:
    static nc::NdArray<double> inline forward(const nc::NdArray<double>& x){
        return x;
    }

    static nc::NdArray<double> inline derivative(const nc::NdArray<double>& x){
        nc::NdArray<double> res(x.shape());
        res.ones();
        return res;
    }
};


class activation_softmax{
public:
    static nc::NdArray<double> inline forward(const nc::NdArray<double>& x){
        nc::NdArray<double> exp = nc::exp(x);
        double sum = exp.sum(nc::Axis::NONE)(0,0);
        nc::applyFunction(exp, std::function([sum](double v) -> double {return v / sum;}));
        return exp;
    }

    static nc::NdArray<double> inline derivative(const nc::NdArray<double>& x){
        // nc::NdArray<double> res = x;
        // nc::applyFunction(res, std::function([](double v) -> double {return v * (1.0 - v);}));
        // return res;
        return x;
    }
};


class activation_sigmoid{
public:
    static nc::NdArray<double> inline forward(const nc::NdArray<double>& x){
        nc::NdArray<double> exp = nc::exp(-x);
        nc::applyFunction(exp, std::function([](double v) -> double {return 1.0 / (1.0 + v);}));
        return exp;
    }

    static nc::NdArray<double> inline derivative(const nc::NdArray<double>& x){
        nc::NdArray<double> res = nc::exp(-x);
        nc::applyFunction(res, std::function([](double v) -> double {return v / ((1 + v) * (1 + v));}));
        return res;
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
        this->x = x;
        nc::NdArray<double> res = nc::dot(x, W) + b; // (1Xin * inXout) + 1Xout
        t = res; // 1Xout
        res = activation::forward(res);
        return res;
    }

    
    //error 1xout
    virtual nc::NdArray<double> backward(const nc::NdArray<double> error, double learn_rate){

        t = activation::derivative(t);
        nc::NdArray<double> dt = error * t; // 1xout
        nc::NdArray<double> dx = nc::dot(dt, W.transpose()); //1Xout * outXin
        b -= learn_rate * dt;
        W -= learn_rate * nc::dot(x.transpose(), dt); //inX1 * 1Xout
        return dx;
    }
};