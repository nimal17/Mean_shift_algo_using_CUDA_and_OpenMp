#ifndef UTILS_H
#define UTILS_H

#include <array>
#include <algorithm> 
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace mean_shift::cuda::utils {

    template <const size_t N, const size_t D>
    std::array<float, N * D> get_file(const std::string& path, const char delim) {
        assert(std::filesystem::exists(path));
        std::ifstream file(path);
        std::string line;
        std::array<float, N * D> dataStr;
        for (size_t i = 0; i < N; ++i) {
            std::getline(file, line);
            std::stringstream lineStr(line);
            std::string cell;
            for (size_t j = 0; j < D; ++j) {
                std::getline(lineStr, cell, delim);
                dataStr[i * D + j] = std::stof(cell);
            }
        }
        file.close();
        return dataStr;
    }

    // template <const size_t N, const size_t D>
    // void write_csv(const std::array<float, N * D>& data, const std::string& path, const char delim) {
    //     std::ofstream output(path);
    //     for (size_t i = 0; i < N; ++i) {
    //         for (size_t j = 0; j < D - 1; ++j)
    //             output << data[i * D + j] << delim;
    //         output << data[i * D + D - 1] << '\n';
    //     }
    //     output.close();
    //     return;
    // }

    // template <typename T, const size_t K>
    // void write_csv(const std::array<T, K>& data, const std::string& path, const char delim) {
    //     std::ofstream output(path);
    //     for (size_t i = 0; i < K; ++i) {
    //         output << data[i] << '\n';
    //     }
    //     output.close();
    //     return;
    // }

    template <const size_t N, const size_t D>
    void print_output(const std::array<float, N * D>& data) {
        for (const auto& c : data) {
            for (int i = 0; i < D; i++)
                std::cout << c << ' ';
            std::cout << '\n';
        }
        return;
    }

    template <typename T, const size_t M>
    void print_output(const std::array<T, M>& data) {
        for (const auto& c : data)
                std::cout << c << '\n';
        return;
    }

    template <const size_t D>
    void print_output(const std::vector<std::array<float, D>>& data) {
        for (const auto& c : data) {
            for (int i = 0; i < D; i++)
                std::cout << c[i] << ' ';
            std::cout << '\n';
        }
        return;
    }

    void data_info(const std::string PATH_TO_DATA, 
                    const size_t N, 
                    const size_t D) {
        std::cout << "\nDATASET:    " << PATH_TO_DATA << '\n';
        std::cout << "NUM POINTS: " << N << '\n';
        std::cout << "DIMENSION:  " << D << '\n';
        return;
    }



    void swap_func(float* &a, float* &b){
        float *temp = a;
        a = b;
        b = temp;
        return;
    }

    template <const size_t N, const size_t D>
    std::vector<std::array<float, D>> conv_centroid(std::array<float, N * D>& data, const float min_distance) {
        std::vector<std::array<float, D>> centroidArr;
        centroidArr.reserve(4);
        std::array<float, D> c1;
        for (size_t j = 0; j < D; ++j) {
            c1[j] = data[j];
        }
        centroidArr.emplace_back(c1);
        for (size_t i = 0; i < N; ++i) {
            bool atleastOne = false;
            for (const auto& c : centroidArr) {
                float dist = 0;
                for (size_t j = 0; j < D; ++j) {
                    dist += ((data[i * D + j] - c[j])*(data[i * D + j] - c[j]));
                }
                if (dist <= min_distance) {
                    atleastOne = true;
                }
            }
            if (not atleastOne) {
                std::array<float, D> centroid;
                for (size_t j = 0; j < D; ++j) {
                    centroid[j] = data[i * D + j];
                }
                centroidArr.emplace_back(centroid);
            }
        }
        return centroidArr;
    }

    }

#endif