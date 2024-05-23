#include <NumCpp.hpp>
#include <vector>
#ifndef LAYER_INTERFACE
#define LAYER_INTERFACE
class activation_relu{
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
        return nc::ones<double>(x.shape());
    }
};


class activation_softmax{
public:
    static nc::NdArray<double> inline forward(const nc::NdArray<double>& x){
        nc::NdArray<double> exp = nc::exp(x - x.max()(0,0));      
        return exp / exp.sum()(0,0);
    }

    static nc::NdArray<double> inline derivative(const nc::NdArray<double>& x){
        return nc::ones<double>(x.shape());
    }
};

class layer_interface{
public:
    virtual std::vector<nc::NdArray<double>> forward(const std::vector<nc::NdArray<double>>& x) = 0;

    virtual std::vector<nc::NdArray<double>> backward(const std::vector<nc::NdArray<double>>& error, double learn_rate) = 0;
};

#endif