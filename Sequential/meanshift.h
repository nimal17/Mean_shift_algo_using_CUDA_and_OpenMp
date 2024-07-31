#ifndef MEAN_SHIFT_H
#define MEAN_SHIFT_H

#include <algorithm>
#include <cmath>
#include "container.h"
#include "container2.h"
#include <iostream>
#include "utils.h"

namespace mean_shift {

    namespace seq {

        template <typename T, const size_t N, const size_t D>
        std::vector<vec<T, D>> cluster(mat<T, N, D>& data, 
                                              const size_t niter, 
                                              const float bandwidth,
                                              const float radius,
                                              const float minDist, 
                                              const double eps) {
            const float double_sqr_bdw = 2 * bandwidth * bandwidth;
            vec<bool, N> has_stopped {false};
            std::vector<vec<T, D>> centroids;
            mat<T, N, D> nData;
            for (size_t i = 0; i < niter; ++i) {
                for (size_t p = 0; p < N; ++p) {
                    if (has_stopped[p]) {
                            if ((centroids.size() == 0) || (check_centroid(centroids, data[p], minDist))) {
                                centroids.emplace_back(data[p]);
                            }
                        continue;
                    }
                    vec<T, D> nPost {};
                    float sumOfweights = 0.;
                    for (size_t q = 0; q < N; ++q) {
                        double dist = distance_helper(data[p], data[q]);
                        if (dist <= radius) {
                            float gaussian = std::exp(- dist / double_sqr_bdw);
                            nPost = nPost + data[q] * gaussian;
                            sumOfweights += gaussian;
                        }
                    }
                    nPost = nPost / sumOfweights;
                    double shift = distance_helper(data[p], nPost);
                    if (shift <= eps) {
                        has_stopped[p] = true;
                    }
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
                                              const float minDist) {
            const float double_sqr_bdw = 2 * bandwidth * bandwidth;
            mat<T, N, D> nData;
            for (size_t i = 0; i < niter; ++i) {
                for (size_t p = 0; p < N; ++p) {
                    vec<T, D> nPost {};
                    float sumOfweights = 0.;
                    for (size_t q = 0; q < N; ++q) {
                        double dist = distance_helper(data[p], data[q]);
                        if (dist <= radius) {
                            float gaussian = std::exp(- dist / double_sqr_bdw);
                            nPost = nPost + data[q] * gaussian;
                            sumOfweights += gaussian;
                        }
                    }
                    nData[p] = nPost / sumOfweights;
                }
                data = nData;
            }
            return conv_centroids(data, minDist);
        }

    } // namespace seq

} // namespace mean_shift

#endif