#include <NumCpp.hpp>


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
                res(matrix.numRows() - i - 1, matrix.numCols() - j - 1) = matrix(i, j);
            }
        }
        return res;
    }

class activation_softmax{
public:
    static nc::NdArray<double> inline forward(const nc::NdArray<double>& x){
        nc::NdArray<double> exp = nc::exp(x);      
        return exp / exp.sum()(0,0);
    }

    static nc::NdArray<double> inline derivative(const nc::NdArray<double>& x){
        return nc::ones<double>(x.shape());
    }
};

    

int main(){
    nc::NdArray<double> test = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    test.reshape(3,3);
    std::cout << test;
    nc::NdArray<double> kernel = {1,1, 1, 1, 1, 1, 1, 1, 1};
    kernel.reshape(3,3);
    std::cout << applyCorrFull(test, kernel); 
    std::cout << turnMatrix(test);
    test = {1, 2, 3, 4 };
    test.reshape(2,2);
    nc::NdArray<double> res(4,4);
    std::cout <<test;
    for (int y = 0; y < 4; y++){
                int in_y = y / 2;
                for (int x = 0; x < 4; x++){
                    res(y, x) = test(in_y, x / 2) / 4.0;
                }
            }
    std::cout << res;
    std::cout << activation_softmax::forward(test);
    test = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    test.reshape(3,3);
    std::vector<nc::NdArray<double>> vec = {test, test};
    nc::NdArray<double> in_res(1, 2 * 3 * 3);
        for (int i = 0; i < 2; i++){
            for (int y = 0; y < 3; y++){
                for (int x = 0; x < 3; x++){
                    int index = i * (3 * 3) + y * 3 + x;
                    in_res(0, index) = vec[i](y, x);
                }
            }
        }
    std::vector<nc::NdArray<double>> test_o;
    for (int i = 0; i < 2; i++){
            nc::NdArray<double> in(3, 3);
            for (int y = 0; y < 3; y++){
                for (int x = 0; x < 3; x++){
                    int index = i * (3 * 3) + y * 3 + x;
                    in(y, x) = in_res(0, index);
                }
            }
            test_o.emplace_back(std::move(in));
        }    
    std::cout << in_res;    
    std::cout << test_o[0];
    std::cout << test_o[1];
    std::cout << nc::pad<double>(test, 2, 0.0);
}