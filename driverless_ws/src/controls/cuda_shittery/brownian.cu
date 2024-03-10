#include <iostream>

#include "cuda_utils.cuh"


// ***** CONFIG *****

constexpr unsigned long long seed = 0;

constexpr size_t action_dims = 5;
constexpr size_t num_timesteps = 128;
constexpr size_t num_samples = 1024;

constexpr curandRngType_t rng_type = CURAND_RNG_PSEUDO_MTGP32;

constexpr uint3 gen_perturb_incrs_block_dims = {1, 64, action_dims};

// derived constexprs
constexpr size_t num_perturbs = num_samples * num_timesteps * action_dims;
constexpr size_t gen_perturbs_block_size =
    gen_perturb_incrs_block_dims.x * gen_perturb_incrs_block_dims.y * gen_perturb_incrs_block_dims.z;
constexpr dim3 perturbs_dims = {num_samples, num_timesteps, action_dims};
constexpr dim3 gen_perturb_incrs_grid_dims = {
    num_samples / gen_perturb_incrs_block_dims.x,
    num_timesteps / gen_perturb_incrs_block_dims.y,
    action_dims / gen_perturb_incrs_block_dims.z
};



// ***** DEVICE GLOBALS *****

__constant__ float perturbs_incr_std[] = {
        1, 0, 0, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 10
};


// ***** DEVICE FUNCS *****

__device__ float dot(const float* v1, const float* v2, size_t n) {
    float res = 0;
    for (size_t i = 0; i < n; i++) {
        res += v1[i] * v2[i];
    }
    return res;
}


// ***** KERNELS *****

/**
 * @brief Generate perturbation increments from standard normals, in-place.
 *        Multilies each action (last axis) in `inout` by `perturbations_incr_std` to generate
 *        Normal increments with `perturbations_incr_std` as the standard deviation matrix
 *
 * @param[in,out] inout Unpitched num_samples x num_timesteps x action_dims tensor. Input standard normals, output
 *                      perturbation increments
 */
__global__ void gen_perturb_incrs(float* inout) {

    //
    __shared__ float shared_perturbs_incr_std[action_dims * action_dims];
    __shared__ float shared_std_normals_block[gen_perturbs_block_size];

    const uint3 grid_idx = total_idx(blockDim, blockIdx, threadIdx);

    float* const shared_std_normals_elem =
        IDX_3D(shared_std_normals_block, gen_perturb_incrs_block_dims, threadIdx);

    float* const inout_elem =
        IDX_3D(inout, perturbs_dims, grid_idx);


    // copy a block of std_normals to shared memory
    *shared_std_normals_elem = *inout_elem;

    // copy perturbs standard dev to shared memory
    thread_parallel_memcpy(
        shared_perturbs_incr_std, perturbs_incr_std,
        action_dims * action_dims,
        blockDim, threadIdx
    );

    __syncthreads();

    float* const perturb_incr_std_row = &shared_perturbs_incr_std[threadIdx.z * action_dims];
    float* const std_normals_action = IDX_3D(shared_std_normals_block, gen_perturb_incrs_block_dims, dim3(threadIdx.x, threadIdx.y, 0));

    *inout_elem = dot(perturb_incr_std_row, std_normals_action, action_dims);
}


// ***** HOST FUNCS *****

cudaTensor3D<float> alloc_perturbs() {
    cudaTensor3D<float> perturbs;
    CUDA_CALL(alloc_tensor3D_unpitched(perturbs, perturbs_dims));
    return perturbs;
}

curandGenerator_t alloc_rng() {
    curandGenerator_t rng;
    CURAND_CALL(curandCreateGenerator(&rng, rng_type));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rng, seed));
    return rng;
}

float* alloc_std_normals_tmp_space() {
    float* ptr;
    CUDA_CALL(cudaMalloc(&ptr, num_perturbs * sizeof(float)));
    return ptr;
}

void gen_std_normals(curandGenerator_t rng, const cudaTensor3D<float>& std_normals, float* tmp_space) {
    CURAND_CALL(curandGenerateNormal(rng, tmp_space, num_perturbs, 0, 1));

    cudaMemcpy3DParms memcpy_params {};
    memcpy_params.extent = std_normals.extent;
    memcpy_params.srcPtr = make_cudaPitchedPtr(tmp_space, sizeof(float) * action_dims, sizeof(float) * action_dims, num_timesteps);
    memcpy_params.dstPtr = std_normals.pitched_ptr;
    memcpy_params.kind = cudaMemcpyDeviceToDevice;

    CUDA_CALL(cudaMemcpy3D(&memcpy_params));
}

void print_perturbs(const cudaTensor3D<float>& perturbs) {
    float* h_perturbations = new float[num_perturbs];

    cudaMemcpy3DParms memcpy_params {};
    memcpy_params.srcPtr = perturbs.pitched_ptr;
    memcpy_params.dstPtr = make_cudaPitchedPtr(h_perturbations, sizeof(float) * action_dims, sizeof(float) * action_dims, num_timesteps);
    memcpy_params.extent = perturbs.extent;
    memcpy_params.kind = cudaMemcpyDeviceToHost;

    CUDA_CALL(cudaMemcpy3D(&memcpy_params));

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_timesteps; j++) {
            std::cout << "{ ";
            for (int k = 0; k < action_dims; k++) {
                std::cout << *IDX_3D(h_perturbations, perturbs_dims, dim3(i, j, k)) << " ";
            }
            std::cout << "} ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    delete [] h_perturbations;
}

int main() {
    cudaTensor3D<float> perturbs = alloc_perturbs();

    curandGenerator_t rng = alloc_rng();
    float* tmp_space = alloc_std_normals_tmp_space();

    int i = 0;
    while (i < 1000000) {
        gen_std_normals(rng, perturbs, tmp_space);

        // print_perturbs(perturbs);

        gen_perturb_incrs<<<gen_perturb_incrs_grid_dims, gen_perturb_incrs_block_dims>>>(
            (float*)perturbs.pitched_ptr.ptr
        );

        // print_perturbs(perturbs);

        cudaDeviceSynchronize();

        if (i % 1000 == 0) {
            std::cout << i << std::endl;
        }

        i++;
    }

    CURAND_CALL(curandDestroyGenerator(rng));

    CUDA_CALL(cudaFree(tmp_space));
    CUDA_CALL(cudaFree(perturbs.pitched_ptr.ptr));
}