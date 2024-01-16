#include <iostream>
#include <curand.h>

#include "cuda_utils.cuh"


constexpr size_t control_dims = 5;
constexpr size_t num_timesteps = 128;
constexpr size_t num_samples = 1024;
constexpr dim3 pertubation_dims = dim3(control_dims, num_timesteps, num_samples);
constexpr size_t num_perturbations = pertubation_dims.x * pertubation_dims.y * pertubation_dims.z;

constexpr int gen_disturbances_block_dim = 256;
constexpr int gen_disturbances_grid_dim = num_samples / gen_disturbances_block_dim;

constexpr curandRngType_t rng_type = CURAND_RNG_PSEUDO_MTGP32;


// ***** Device Memory *****

__constant__ float d_disturbance_std[control_dims * control_dims];


// ***** Device Helpers *****

__device__ void square_matmul_vec_unpitched(int dims, float* mat, float* vec, float* res) {
    for (int i = 0; i < dims; i++) {
        res[i] = 0;
        for (int j = 0; j < dims; j++) {
            res[i] += mat[i * dims + j] * vec[j];
        }
    }
}


// ***** Kernels *****

__global__ void gen_perturbations(float* perturbations, const size_t pitch) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float res[control_dims];
    
    square_matmul_vec_unpitched(control_dims, d_disturbance_std, 
                                IDX_3D_PITCHED(perturbations, pertubation_dims, dim3(0, 0, tid), pitch),
                                res);

    memcpy(IDX_3D_PITCHED(perturbations, pertubation_dims, dim3(0, 0, tid), pitch), res, sizeof(res));

    for (int j = 1; j < num_timesteps; j++) {
        square_matmul_vec_unpitched(control_dims, d_disturbance_std, 
                                    IDX_3D_PITCHED(perturbations, pertubation_dims, dim3(0, j, tid), pitch),
                                    res);
                                    
        for (int k = 0; k < control_dims; k++) {
            *IDX_3D_PITCHED(perturbations, pertubation_dims, dim3(k, j, tid), pitch) =
                *IDX_3D_PITCHED(perturbations, pertubation_dims, dim3(k, j - 1, tid), pitch) + res[k];
        }
    }
}


// ***** Host *****

void set_disturbance_std() {
    float h_disturbance_std[] = {
        1, 0, 0, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 10
    };

    CUDA_CALL(cudaMemcpyToSymbol(d_disturbance_std, h_disturbance_std, sizeof(h_disturbance_std)));
}

cudaTensor3D alloc_perturbations() {
    cudaTensor3D perturbations;

    perturbations.extent = make_cudaExtent(control_dims * sizeof(float), num_timesteps, num_samples);
    CUDA_CALL(cudaMalloc3D(&perturbations.pitched_ptr, perturbations.extent));

    return perturbations;
}

void print_pertrubations(const cudaTensor3D& perturbations) {
    float* h_perturbations = new float[num_perturbations];

    cudaMemcpy3DParms memcpy_params {};
    memcpy_params.srcPtr = perturbations.pitched_ptr;
    memcpy_params.dstPtr = make_cudaPitchedPtr(h_perturbations, sizeof(float) * control_dims, sizeof(float) * control_dims, num_timesteps);
    memcpy_params.extent = perturbations.extent;
    memcpy_params.kind = cudaMemcpyDeviceToHost;

    CUDA_CALL(cudaMemcpy3D(&memcpy_params));

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_timesteps; j++) {
            std::cout << "{ ";
            for (int k = 0; k < control_dims; k++) {
                std::cout << *IDX_3D(h_perturbations, pertubation_dims, dim3(k, j, i)) << " ";
            }
            std::cout << "} ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    delete [] h_perturbations;
}

curandGenerator_t alloc_rng() {
    curandGenerator_t rng;
    CURAND_CALL(curandCreateGenerator(&rng, rng_type));
    return rng;
}

float* alloc_normals_tmp_space() {
    float* ptr;
    CUDA_CALL(cudaMalloc(&ptr, num_perturbations * sizeof(float)));
    return ptr;
}

void gen_normals(curandGenerator_t rng, const cudaTensor3D& perturbations, float* tmp_space) {
    CURAND_CALL(curandGenerateNormal(rng, tmp_space, num_perturbations, 0, 1));

    cudaMemcpy3DParms memcpy_params {};
    memcpy_params.extent = perturbations.extent;
    memcpy_params.srcPtr = make_cudaPitchedPtr(tmp_space, sizeof(float) * control_dims, sizeof(float) * control_dims, num_timesteps);
    memcpy_params.dstPtr = perturbations.pitched_ptr;
    memcpy_params.kind = cudaMemcpyDeviceToDevice;

    CUDA_CALL(cudaMemcpy3D(&memcpy_params));
}

int main() {
    set_disturbance_std();

    cudaTensor3D perturbations = alloc_perturbations();
    curandGenerator_t rng = alloc_rng();
    float* normals_tmp_space = alloc_normals_tmp_space();

    int i = 0;
    while (i < 1000000) {
        gen_normals(rng, perturbations, normals_tmp_space);

        // print_pertrubations(perturbations);

        gen_perturbations<<<gen_disturbances_grid_dim, gen_disturbances_block_dim>>>(
            static_cast<float*>(perturbations.pitched_ptr.ptr),
            perturbations.pitched_ptr.pitch
        );

        // print_pertrubations(perturbations);

        // std::cout << "-------------" << std::endl;

        CUDA_CALL(cudaStreamSynchronize(nullptr));

        if (i % 1000 == 0) {
            std::cout << i << std::endl;
        }

        i++;
    };

    curandDestroyGenerator(rng);

    cudaFree(perturbations.pitched_ptr.ptr);
    cudaFree(normals_tmp_space);
}