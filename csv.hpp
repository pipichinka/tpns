#include "NumCpp.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>



class csv{
    std::vector<std::string> head;
    std::vector<nc::NdArray<double>> data;


public:
    csv(const std::string& file){
        std::ifstream in_file(file);
        if (in_file.bad()){
            throw std::runtime_error("can't open file");
        }
        char line[1024];
        in_file.getline(line, 1023);
        char* current_line_start = line;
        char* line_end = line + strlen(line);
        while (current_line_start < line_end){
            char* sep_pos = strchrnul(current_line_start, ',');
            std::string value(current_line_start, 0, sep_pos - current_line_start);
            head.push_back(value);
            current_line_start = sep_pos + 1;
        }

        while (!in_file.eof()){
            in_file.getline(line, 1023);
            char* current_line_start = line;
            std::vector<double> current_iteration_vector;
            char* line_end = line + strlen(line);
            if (line == line_end){
                continue;
            }
            while (current_line_start < line_end){
                char* sep_pos = strchrnul(current_line_start, ',');
                std::string value(current_line_start, 0, sep_pos - current_line_start);
                std::stod(value);
                current_line_start = sep_pos + 1;
                current_iteration_vector.push_back(std::stod(value));
            }
            data.emplace_back(current_iteration_vector);
        }

    }

    const std::vector<std::string>& get_head(){
        return head;
    }

    const std::vector<nc::NdArray<double>>& get_data(){
        return data;
    }

    std::vector<nc::NdArray<double>> get_data_by_names(const std::vector<std::string>& names){
        std::vector<unsigned int> indexes;
        for (std::string name: names){
            for (int i = 0; i < head.size(); ++i){
                if (name == head[i]){
                    indexes.push_back(i);
                }
            }
        }
        
        std::vector<nc::NdArray<double>> result;
        nc::NdArray<unsigned int> nc_indexes(indexes);
        for (nc::NdArray<double>& data_i: data){
            result.emplace_back(data_i.getByIndices(nc_indexes));
        }
        return result;
    }

    ~csv() = default;    

};