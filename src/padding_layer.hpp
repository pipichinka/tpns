#include <NumCpp.hpp>
#include <vector>
#include "layer_interface.hpp"

class padding_layer: public layer_interface{
    int width;

public:
    padding_layer(int width):width(width){

    }


    virtual std::vector<nc::NdArray<double>> forward(const std::vector<nc::NdArray<double>>& x){
        std::vector<nc::NdArray<double>> res;
        for (const auto& array: x){
            res.emplace_back(nc::pad<double>(array, width, 0.0));
        }
        return res;
    }


    virtual std::vector<nc::NdArray<double>> backward(const std::vector<nc::NdArray<double>>& error, double learn_rate){
        return error;//we don't use it
    }

};