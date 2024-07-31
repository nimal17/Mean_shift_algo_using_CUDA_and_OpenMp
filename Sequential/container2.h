#ifndef CONTAINER_IO_H
#define CONTAINER_IO_H

#include <cassert>
#include "container.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace mean_shift::container {

    template <typename T, const size_t N, const size_t D>
    mat<T, N, D> get_file(const std::string& path, const char deLim) {
        assert(std::filesystem::exists(path));
        std::ifstream file(path);
        std::string line;
        mat<T, N, D> dataStr;
        for (size_t i = 0; i < N; ++i) {
            std::getline(file, line);
            std::stringstream lineStr(line);
            std::string cell;
            vec<T, D> point;
            for (size_t j = 0; j < D; ++j) {
                std::getline(lineStr, cell, deLim);

                point[j] = static_cast<T>(std::stod(cell));
            }
            dataStr[i] = point;
        }
        file.close();
        return dataStr;
    }

    


    template <typename T, const size_t D>
    void print_vec(const vec<T, D>& vector) {
        for (auto v : vector)
            std::cout << v << ' ';
        std::cout << '\n';
        return;
    }

    

    template <typename T, const size_t D>
    void print_mat(const std::vector<vec<T, D>>& matrix) {
        for (const vec<T, D>& vector : matrix)
            print_vec<T, D>(vector);
        std::cout << '\n';
        return;
    }

} // namespace container

#endif