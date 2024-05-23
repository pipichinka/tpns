#include <NumCpp.hpp>
#include "csv.hpp"
#include "perceptron.hpp"
#include "rnn_layer.hpp"

std::vector<std::string> parse_in_comma_sep(char* in){
    std::vector<std::string> result;
    char* current_line_start = in;
    char* line_end = in + strlen(in);
    while (current_line_start < line_end){
        char* sep_pos = strchrnul(current_line_start, ',');
        std::string value(current_line_start, 0, sep_pos - current_line_start);
        result.push_back(value);
        current_line_start = sep_pos + 1;
    }
    return result;
}


std::vector<int> transform_s_to_i_vec(const std::vector<std::string>& in){
    std::vector<int> res;
    for (const std::string& s: in){
        res.push_back(std::stoi(s));
    }
    return res;
}


std::vector<layer_interface*> make_layers(const std::vector<int> in_layers, int train_layer_size, int target_layer_size){
    std::vector<layer_interface*> res;
    if (in_layers.size() == 0){
        res.emplace_back(new layer<activation_sigmoid>(train_layer_size, target_layer_size));
        return res;
    }
    res.emplace_back(new layer<activation_relu>(train_layer_size, in_layers[0]));
    for (int i = 0; i < in_layers.size() - 1; i++){
        res.emplace_back(new layer<activation_relu>(in_layers[i], in_layers[i + 1]));
    }
    res.emplace_back(new layer<activation_softmax>(in_layers.back(), target_layer_size));
    return res;
}


nc::NdArray<int> make_cat(const nc::NdArray<double>& in, double threshold){
    nc::NdArray<int> res(in.shape());
    for (int i = 0; i < in.shape().cols; i++){
        if (in(0, i) < threshold){
            res(0, i) = 0;
        } else {
            res(0, i) = 1;
        }
    }
    return res;
}


void count_roc(perceptron& p, std::vector<nc::NdArray<double>>& in_data, std::vector<nc::NdArray<double>>& out_data, const std::vector<std::string>& out_data_columns){
    std::vector<double> tpr_vector;
    std::vector<double> fpr_vector;
    
    for (double threshold = -0.05; threshold <= 1.5; threshold += 0.05){
        nc::NdArray<int> tp = nc::zeros<int>(out_data[0].shape());
        nc::NdArray<int> tn = tp;
        nc::NdArray<int> fp = tp;
        nc::NdArray<int> fn = tp;
        for (size_t i = in_data.size() * 0.8; i < in_data.size(); i++){
            nc::NdArray<int> res =make_cat(p.solve(in_data[i]), threshold);
            for (int j = 0; j < out_data[0].shape().cols; j++){
                if (res(0, j) == 1 && out_data[i](0, j) == 1){
                    tp(0, j) += 1;
                }
                if (res(0, j) == 0 && out_data[i](0, j) == 0){
                    tn(0, j) += 1;
                }
                if (res(0, j) == 1 && out_data[i](0, j) == 0){
                    fp(0, j) += 1;
                }
                if (res(0, j) == 0 && out_data[i](0, j) == 1){
                    fn(0, j) += 1;
                }
            }
        }
        for (int j = 0; j < out_data[0].shape().cols; j++){

            double tpr = ((double) tp(0, j)) / ((double) tp(0, j) + fn(0, j));
            tpr_vector.push_back(tpr);
            //std::cout << "tpr of " << out_data_columns[j] << ": ["  << threshold << ", "<< tpr << "]" << std::endl;

            double fpr = (double) fp(0, j) / ((double) fp(0, j) + tn(0, j));
            fpr_vector.push_back(fpr);
            //std::cout << "fpr " << out_data_columns[j] << ": ["  << threshold << ", "<< fpr << "]" << std::endl;
        }
    }
    for (double v: tpr_vector){
        std::cout << v  << ", ";
    }
    std::cout << std::endl;
    for (double v: fpr_vector){
        std::cout << v  << ", ";
    }
    std::cout << std::endl;
}


void test_perspetron(perceptron& p, std::vector<nc::NdArray<double>>& in_data, std::vector<nc::NdArray<double>>& out_data, const std::vector<std::string>& out_data_columns){

    nc::NdArray<int> tp = nc::zeros<int>(out_data[0].shape());
    nc::NdArray<int> tn = tp;
    nc::NdArray<int> fp = tp;
    nc::NdArray<int> fn = tp;
    for (size_t i = in_data.size() * 0.8; i < in_data.size(); i++){
        nc::NdArray<int> res = make_cat(p.solve(in_data[i]), 0.5);
        for (int j = 0; j < out_data[0].shape().cols; j++){
            if (res(0, j) == 1 && out_data[i](0, j) == 1){
                tp(0, j) += 1;
            }
            if (res(0, j) == 0 && out_data[i](0, j) == 0){
                tn(0, j) += 1;
            }
            if (res(0, j) == 1 && out_data[i](0, j) == 0){
                fp(0, j) += 1;
            }
            if (res(0, j) == 0 && out_data[i](0, j) == 1){
                fn(0, j) += 1;
            }
        }
    }

    std:: cout << tp << tn << fp << fn;
    for (int j = 0; j < out_data[0].shape().cols; j++){
        double accuracy = ((double) tp(0, j) + tn(0, j)) / ((double) tp(0, j) + tn(0, j) + fp(0, j) + fn(0, j));
        std::cout << "acurracy of " << out_data_columns[j] << ": " << accuracy << std::endl;
        
        double precision = ((double) tp(0, j)) / ((double) tp(0, j) + fp(0, j));
        std::cout << "precision of " << out_data_columns[j] << ": " <<  precision << std::endl;

        double recall = ((double) tp(0, j)) / ((double) tp(0, j) + fn(0, j));
        std::cout << "recall of " << out_data_columns[j] << ": " << recall << std::endl;

        double f_score = 2.0 * (precision * recall) / (precision + recall);
        std::cout << "f score " << out_data_columns[j] << ": " << f_score << std::endl;
    }

    std::cout << std::endl << std::endl;
    count_roc(p, in_data, out_data, out_data_columns);


}


// file train(comma sep list) target(comma sep list) epoch
int main(int argc, char** argv){

    if (argc != 5){
        std::cerr << "usage: sv_file train(comma sep list) target(comma sep list) epoch\n";
        return 1;
    }

    csv* file = new csv(argv[1]);

    std::vector<std::string> train_columns(parse_in_comma_sep(argv[2]));
    for (std::string& column: train_columns){
        std::cout << column << " ";
    }
    std::cout << std::endl;

    std::vector<std::string> target_columns(parse_in_comma_sep(argv[3]));
    for (std::string& column: target_columns){
        std::cout << column << " ";
    }
    std::cout << std::endl;

    std::vector<nc::NdArray<double>> train_data(file->get_data_by_names(train_columns));
    for (int i =0; i < 5; i++){
        std::cout << train_data[i] << " ";
    }
    std::cout << std::endl;
    std::vector<nc::NdArray<double>> target_data(file->get_data_by_names(target_columns));
    for (int i =0; i < 5; i++){
        std::cout << target_data[i] << " ";
    }
    std::cout << std::endl;
    std::vector<layer_interface*> layers(make_layers(
        transform_s_to_i_vec({}), 
        train_columns.size(), 
        target_columns.size()));

    error_counter_interface* error_counter = new common_error_counter();
    perceptron p(layers, error_counter);
    p.train(train_data, target_data, train_data.size() * 0.8 , std::stoi(argv[4]), 0.1);
    test_perspetron(p, train_data, target_data, target_columns);
}