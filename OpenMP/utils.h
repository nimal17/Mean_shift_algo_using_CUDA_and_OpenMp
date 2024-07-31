#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <numeric> // iota

namespace mean_shift {

    template<typename T, const size_t D>
    double distance_helper(const vec<T, D>& p, const vec<T, D>& q) {
        double sum = 0.0;
        for (size_t i = 0; i < D; ++i)
            sum += ((p[i] - q[i]) * (p[i] - q[i]));
        return sum;
    }

    template <typename T, const size_t D>
    bool check_centroid(std::vector<vec<T, D>>& currentCentroids, const vec<T, D>& point, const double epsCluster ) {
        return std::none_of(currentCentroids.begin(), 
                            currentCentroids.end(), 
                            [&](const auto& c) {return distance_helper(c, point) <= epsCluster;});
    }

    template<typename T, const size_t M, const size_t D>
    bool close_actual(const std::vector<vec<T, D>>& centroids, const mat<T, M, D>& real, const double epsConv) {
        vec<bool, M> areClose {false};
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < M; ++j) {
                if (distance_helper(centroids[i], real[j]) <= epsConv)
                    areClose[i] = true;
            }
        }
        return std::all_of(areClose.begin(), areClose.end(), [](const bool b){return b;});
    }

    template <typename T, const size_t N, const size_t D>
    std::vector<vec<T, D>> conv_centroids(mat<T, N, D>& data, const float minDist) {
        std::vector<vec<T, D>> centroids = {data[0]};
        for (const auto& p : data) {
            if (check_centroid(centroids, p, minDist))
                centroids.emplace_back(p);
        }
        return centroids;
    }



}

#endif
