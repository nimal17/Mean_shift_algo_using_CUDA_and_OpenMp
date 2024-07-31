#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

namespace mean_shift::cuda {

    constexpr float radius = 60;
    constexpr float sigma = 4;
    constexpr float sigma2Sqr = (2 * sigma * sigma);
    constexpr float minDist = 60;
    constexpr size_t numIterations = 50;
    const std::string filePath = "../datasets/3d/10000_samples_3_centers/data.csv";
    constexpr int N = 10000;
    constexpr int D = 3;
    
    constexpr int threads = 64;
    constexpr int blocks = (N + threads - 1) / threads;
    constexpr int tileWid = threads;

}

#endif