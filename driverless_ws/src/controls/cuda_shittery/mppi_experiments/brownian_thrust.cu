#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include "cuda_utils.cuh"

#include <iostream>
#include <cmath>

#include "config.cuh"
#include "cuda_utils.cuh"


// ***** DEVICE FUNCS *****

template<typename T>
__host__ __device__ T dot(const T* a, const T* b, size_t n) {
    T res {};
    for (size_t i = 0; i < n; i++) {
        res += a[i] * b[i];
    }
    return res;
}

struct TransformStdNormal {
    thrust::device_ptr<float> std_normals;

    explicit TransformStdNormal(thrust::device_ptr<float> std_normals)
        : std_normals {std_normals} { }

    __host__ __device__ void operator() (size_t idx) const {
        const size_t action_idx = (idx / action_dims) * action_dims;
        const size_t row_idx = idx % action_dims * action_dims;

        const float res = dot<float>(&perturbs_incr_std[row_idx], &std_normals.get()[action_idx], action_dims);
        std_normals[idx] = res * sqrt_timestep;
    }
};


// **** HOST FUNCS ****

curandGenerator_t alloc_rng() {
    curandGenerator_t rng;
    CURAND_CALL(curandCreateGenerator(&rng, rng_type));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rng, seed));
    return rng;
}

void gen_normals(thrust::device_ptr<float> normal, curandGenerator_t rng) {
    CURAND_CALL(curandGenerateNormal(rng, normal.get(), num_perturbs, 0, 1));
}

void prefix_scan(thrust::device_ptr<float> normals) {
    auto actions = thrust::device_pointer_cast((Action*)normals.get());
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), DivBy<num_timesteps> {});

    thrust::inclusive_scan_by_key(keys, keys + num_samples * num_timesteps,
                                  actions, actions,
                                  Equal<size_t> {}, AddActions {});
}

int main() {
    curandGenerator_t rng = alloc_rng();
    float* normal_raw;
    CUDA_CALL(cudaMalloc(&normal_raw, sizeof(float) * num_perturbs));

    thrust::device_ptr<float> normal = thrust::device_pointer_cast(normal_raw);

    for (int i = 0; i < 1000000; i++) {
        gen_normals(normal, rng);

        // print_tensor_3D(normal, brownian_dims);

        thrust::counting_iterator<size_t> indices {0};
        thrust::for_each(indices, indices + num_perturbs, TransformStdNormal {normal});

        // print_tensor_3D(normal, brownian_dims);

        prefix_scan(normal);

        // print_tensor_3D(normal, brownian_dims);

        if (i % 1000 == 0) {
            std::cout << i << std::endl;
        }
    }

    CURAND_CALL(curandDestroyGenerator(rng));
    CUDA_CALL(cudaFree(normal.get()));
}