#include <chrono>
#include "constants.h"
#include <cuda.h>
#include <iostream>
#include "utils.h"

namespace mean_shift::cuda {

    __global__ void performMeanShift(float *points, float *updatedPoints) {
        size_t threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (threadId < N) {
            size_t currentRow = threadId * D;
            float newPosition[D] = {0.};
            float totalWeight = 0.;
            for (size_t i = 0; i < N; ++i) {
                size_t otherRow = i * D;
                float squaredDistance = 0.;
                for (size_t j = 0; j < D; ++j) {
                    float distanceComponent = points[currentRow + j] - points[otherRow + j];
                    squaredDistance += distanceComponent * distanceComponent;
                }
                if (squaredDistance <= radius) {
                    float weight = expf(-squaredDistance / sigma2Sqr);
                    for (size_t j = 0; j < D; ++j) {
                        newPosition[j] += weight * points[otherRow + j];
                    }
                    totalWeight += weight;
                }
            }
            for (size_t j = 0; j < D; ++j) {
                updatedPoints[currentRow + j] = newPosition[j] / totalWeight;
            }
        }
    }

}

int main() {

    constexpr auto numPoints = mean_shift::cuda::N;
    constexpr auto dimensions = mean_shift::cuda::D;
    constexpr auto numThreads = mean_shift::cuda::threads;
    constexpr auto numBlocks = mean_shift::cuda::blocks;
    const auto dataFilePath = mean_shift::cuda::filePath; 
    
    std::array<float, numPoints * dimensions> currentData = mean_shift::cuda::utils::get_file<numPoints, dimensions>(dataFilePath, ',');
    std::array<float, numPoints * dimensions> nextData {};
    float *deviceCurrentData;
    float *deviceNextData;

    size_t bytesRequired = numPoints * dimensions * sizeof(float);
    cudaMalloc(&deviceCurrentData, bytesRequired);
    cudaMalloc(&deviceNextData, bytesRequired);


    cudaMemcpy(deviceCurrentData, currentData.data(), bytesRequired, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNextData, nextData.data(), bytesRequired, cudaMemcpyHostToDevice);

    auto startTime = std::chrono::system_clock::now();
    for (size_t iteration = 0; iteration < mean_shift::cuda::numIterations; ++iteration) {
        mean_shift::cuda::performMeanShift<<<numBlocks, numThreads>>>(deviceCurrentData, deviceNextData);
        cudaDeviceSynchronize();
        mean_shift::cuda::utils::swap_func(deviceCurrentData, deviceNextData);
    }
    cudaMemcpy(currentData.data(), deviceCurrentData, bytesRequired, cudaMemcpyDeviceToHost);
    auto detectedCentroids = mean_shift::cuda::utils::conv_centroid<numPoints, dimensions>(currentData, mean_shift::cuda::minDist);
    auto endTime = std::chrono::system_clock::now();
    auto durationMS = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "\nTime taken : " << durationMS << " ms\n" << std::endl;

    cudaFree(deviceCurrentData);
    cudaFree(deviceNextData);

    mean_shift::cuda::utils::print_output<dimensions>(detectedCentroids);
    std::cout << "There are " << detectedCentroids.size() << " centroids.\n";

    

    return 0;
}
