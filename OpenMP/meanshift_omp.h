#ifndef MEAN_SHIFT_H
#define MEAN_SHIFT_H

#include <algorithm>
#include <cmath>
#include "container.h"
#include "container2.h"
#include <iostream>
#include "utils.h"


namespace mean_shift {

    namespace omp {

        template <typename T, const size_t N, const size_t D>
        std::vector<vec<T, D>> cluster(mat<T, N, D>& data, 
                                              const size_t niter, 
                                              const float bandwidth,
                                              const float radius,
                                              const float min_distance, 
                                              const double eps) {
            const float bandwidthDsqr = 2 * bandwidth * bandwidth;
            vec<bool, N> has_stopped {false};
            std::vector<vec<T, D>> centroids;
            mat<T, N, D> nData;
            for (size_t i = 0; i < niter; ++i) {
                #pragma omp parallel for default(none) \
                shared(data, niter, bandwidth, eps, radius, bandwidthDsqr, has_stopped, centroids, nData, min_distance) \
                schedule(dynamic)
                for (size_t p = 0; p < N; ++p) {
                    if (has_stopped[p]) {
                        #pragma omp critical
                        {
                            if ((centroids.size() == 0) || (check_centroid(centroids, data[p], min_distance))) {
                                centroids.emplace_back(data[p]);
                            }
                        }
                        continue;
                    }
                    vec<T, D> nPost {};
                    float sumofWeights = 0.;
                    for (size_t q = 0; q < N; ++q) {
                        double dist = distance_helper(data[p], data[q]);
                        if (dist <= radius) {
                            float gaussian = std::exp(- dist / bandwidthDsqr);
                            nPost = nPost + data[q] * gaussian;
                            sumofWeights += gaussian;
                        }
                    }
                    nPost = nPost / sumofWeights;
                    double shift = distance_helper(data[p], nPost);
                    if (shift <= eps) {
                        #pragma omp atomic write
                        has_stopped[p] = true;
                    }
                    #pragma omp critical
                    nData[p] = nPost;
                }
                data = nData;
                if (std::all_of(has_stopped.begin(), has_stopped.end(), [](bool b) {return b;})) {
                    std::cout << "With eps = " << eps << " took " << i << " iterations!\n";
                    return centroids;
                }
            }
            return centroids;
        }

        template <typename T, const size_t N, const size_t D>
        std::vector<vec<T, D>> cluster(mat<T, N, D>& data, 
                                              const size_t niter, 
                                              const float bandwidth,
                                              const float radius,
                                              const float min_distance) {
            const float bandwidthDsqr = 2 * bandwidth * bandwidth;
            mat<T, N, D> nData;
            for (size_t i = 0; i < niter; ++i) {
                #pragma omp parallel for default(none) \
                shared(data, niter, bandwidth, radius, bandwidthDsqr, nData) \
                schedule(dynamic)
                for (size_t p = 0; p < N; ++p) {
                    vec<T, D> nPost {};
                    float sumofWeights = 0.;
                    for (size_t q = 0; q < N; ++q) {
                        double dist = distance_helper(data[p], data[q]);
                        if (dist <= radius) {
                            float gaussian = std::exp(- dist / bandwidthDsqr);
                            nPost = nPost + data[q] * gaussian;
                            sumofWeights += gaussian;
                        }
                    }
                    #pragma omp critical
                    nData[p] = nPost / sumofWeights;
                }
                data = nData;
            }
            return conv_centroids(data, min_distance);
        }

    } // namespace omp

} // namespace mean_shift

#endif