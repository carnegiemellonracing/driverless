#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include "cuda_utils.cuh"

#include <iostream>
#include <cmath>


// ***** CONFIG *****

constexpr size_t action_dims = 5;
constexpr size_t num_timesteps = 5;
constexpr size_t num_samples = 5;
constexpr size_t num_brownians = action_dims*num_timesteps*num_samples;

constexpr curandRngType_t rng_type = CURAND_RNG_PSEUDO_MTGP32;
constexpr unsigned long long seed = 0;
constexpr float sqrt_timestep = 1.0f;

__constant__ const float perturbs_incr_std[] = {
    1, 0, 0, 0, 0,
    0, 2, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 10
};


// ***** DEVICE FUNCS *****

template<typename T>
__device__ T dot(const T* a, const T* b, size_t n) {
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

    __device__ void operator() (size_t idx) const {
        const size_t action_idx = (idx / action_dims) * action_dims;
        const size_t row_idx = idx % action_dims * action_dims;

        const float res = dot<float>(&perturbs_incr_std[row_idx], &std_normals.get()[action_idx], action_dims);
        std_normals[idx] = res * sqrt_timestep;
    }
};

__device__ static size_t mod_action_dims(size_t i) {
    return i % action_dims;
}


// **** HOST FUNCS ****

curandGenerator_t alloc_rng() {
    curandGenerator_t rng;
    CURAND_CALL(curandCreateGenerator(&rng, rng_type));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rng, seed));
    return rng;
}

thrust::device_ptr<float> gen_normals(curandGenerator_t rng) {
    float * normal_raw_ptr;
    cudaMalloc(&normal_raw_ptr, num_brownians * sizeof(float));

    CURAND_CALL(curandGenerateNormal(rng, normal_raw_ptr, num_brownians, 0, 1));

    return thrust::device_pointer_cast(normal_raw_ptr);
}

void prefix_scan(thrust::device_ptr<float> normals) {
    auto from_zero = thrust::make_counting_iterator<size_t>(0);
    auto sawtooth = thrust::make_transform_iterator(from_zero, mod_action_dims);
    thrust::inclusive_scan_by_key(sawtooth, sawtooth + num_brownians, normals, normals);
}


int main(void) {
    thrust::host_vector<float> H {action_dims*num_timesteps*num_samples};

    curandGenerator_t rng = alloc_rng();
    thrust::device_ptr<float>  normal = gen_normals(rng);

    thrust::counting_iterator<size_t> indices {0};
    thrust::for_each(indices, indices + num_brownians, TransformStdNormal {normal});

    prefix_scan(normal);

    thrust::copy(normal, normal + num_brownians, H.begin());

    for (int i = 0; i < num_brownians; i++) {
        std::cout << H[i] << std::endl;
    }

    CURAND_CALL(curandDestroyGenerator(rng));
    CUDA_CALL(cudaFree(normal.get()));
}