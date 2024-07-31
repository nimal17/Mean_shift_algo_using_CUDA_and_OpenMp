#include <cassert>
#include <chrono>
#include "meanshift_omp.h"
#include "container2.h"

int main() {

    const std::string data_path = "../datasets/3d/10000_samples_3_centers/data.csv";

    const float radius = 60;
    const float minDist = 60;
    const float bandwidth = 3;
    const size_t niter = 50;

    const size_t numPoints = 10000;
    const size_t dim = 3;
    
   
    mean_shift::mat<float, numPoints, dim> data = mean_shift::container::get_file<float, numPoints, dim>(data_path, ',');
    auto start = std::chrono::high_resolution_clock::now();
    const std::vector<mean_shift::vec<float, dim>> centroids = mean_shift::omp::cluster<float, numPoints, dim>(data, niter, bandwidth, radius, minDist);    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration << " ms" << "\n\n";
    mean_shift::container::print_mat(centroids);
    std::cout << "There are " << centroids.size() << " centroids.\n";
        
    return 0;
}