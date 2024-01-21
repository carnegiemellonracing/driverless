#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include "cuda_utils.cuh"

#include <iostream>
#include <cmath>


// ***** CONFIG *****

constexpr size_t action_dims = 5;
constexpr size_t num_timesteps = 3;
constexpr size_t num_samples = 3;
constexpr size_t num_brownians = action_dims*num_timesteps*num_samples;
constexpr dim3 brownian_dims {num_samples, num_timesteps, action_dims};

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

template<size_t k>
struct MultiplyBy {
    __host__ __device__ size_t operator() (size_t i) const {
        return i * k;
    }
};

struct Action {
    float data[action_dims];
};

struct AddActions {
    __host__ __device__ Action operator() (Action a1, Action a2) {
        Action res;
        for (size_t i = 0; i < action_dims; i++) {
            res.data[i] = a1.data[i] + a2.data[i];
        }
        return res;
    }
};

template<size_t k>
struct DivBy {
    __host__ __device__ size_t operator() (size_t i) const {
        return i / k;
    }
};

template<typename T>
struct Equal {
    __host__ __device__ bool operator() (T a, T b) {
        return a == b;
    }
};


// **** HOST FUNCS ****

curandGenerator_t alloc_rng() {
    curandGenerator_t rng;
    CURAND_CALL(curandCreateGenerator(&rng, rng_type));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rng, seed));
    return rng;
}

thrust::device_ptr<float> gen_normals(curandGenerator_t rng) {
    float* normal_raw_ptr;
    cudaMalloc(&normal_raw_ptr, num_brownians * sizeof(float));

    CURAND_CALL(curandGenerateNormal(rng, normal_raw_ptr, num_brownians, 0, 1));

    return thrust::device_pointer_cast(normal_raw_ptr);
}

void prefix_scan(thrust::device_ptr<float> normals) {
    auto actions = thrust::device_pointer_cast((Action*)normals.get());
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), DivBy<num_timesteps> {});

    thrust::inclusive_scan_by_key(keys, keys + num_samples * num_timesteps,
                                  actions, actions,
                                  Equal<size_t> {}, AddActions {});
}

template<typename T>
void print_tensor_3D(T tensor, dim3 dims) {
    for (int i = 0; i < dims.x; i++) {
        for (int j = 0; j < dims.y; j++) {
            std::cout << "{ ";
            for (int k = 0; k < dims.z; k++) {
                std::cout << *IDX_3D(tensor, dims, dim3(i, j, k)) << " ";
            }
            std::cout << "} ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    curandGenerator_t rng = alloc_rng();
    thrust::device_ptr<float> normal = gen_normals(rng);

    print_tensor_3D(normal, brownian_dims);

    thrust::counting_iterator<size_t> indices {0};
    thrust::for_each(indices, indices + num_brownians, TransformStdNormal {normal});

    print_tensor_3D(normal, brownian_dims);

    prefix_scan(normal);

    print_tensor_3D(normal, brownian_dims);

    CURAND_CALL(curandDestroyGenerator(rng));
    CUDA_CALL(cudaFree(normal.get()));
}