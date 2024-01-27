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

__device__ void model(const float state[], const float action[], float state_dot[]) {
   // return the sum of the two vectors
    for (size_t i = 0; i < state_dims; i++) {
        state_dot[i] = state[i] + action[i];
    }
}

__device__ float cost(float state[]) {
    // sum the vector of state
    float sum = 0;
    for (size_t i = 0; i < state_dims; i++) {
        sum += state[i];
    } 
    return sum;
}

struct PopulateCost {
    thrust::device_ptr<float> brownians;
    thrust::device_vector<float> sampled_action_trajectories;
    thrust::device_vector<float> cost_to_gos;
    
    const thrust::device_vector<float> action_trajectory_base;
    const thrust::device_vector<float> curr_state;


    PopulateCost(thrust::device_ptr<float> brownians) 
        : brownians {brownians}, 
          sampled_action_trajectories {sampled_action_trajectories},
          cost_to_gos {cost_to_gos},
          action_trajectory_base {action_trajectory_base},
          curr_state {curr_state} {}

    __device__ void operator() (size_t i) const {
        float j_curr = 0;
        float x_curr[] = currState;
        for (size_t j = 0; j < num_timesteps; j++) {
            const float *u_ij = IDX_3D(sampled_action_trajectories, dim3 {i, j, 0}, dim3 {num_timesteps, action_dims});
            for (size_t k = 0; k < action_dims; k++) {
                const size_t idx = j * action_dims + k;
                u_ij[k] = action_trajectory_base[idx] 
                       + *IDX_3D(brownians, dim3 {i, j, k}, dim3 {num_perturbs, num_timesteps, action_dims});
            }
            x_curr = model(x_curr, u_ij) * timestep; // Euler's method, TODO make better;
            j_curr -= cost(x_curr);
            cost_to_gos[i * num_timesteps + j] = j_curr;
        } 
    }
};



int main() {
    thrust::device_vector<float> brownians_vec {num_perturbs, 1};
    thrust::device_ptr<float> brownians = thrust::device_pointer_cast(brownians_vec);

    thrust::device_vector<float> sampled_action_trajectories {num_perturbs};
    thrust::device_vector<float> cost_to_gos {num_samples * num_timesteps};

    thrust::device_vector<float> action_trajectory_base {num_timesteps * action_dims, 0};
    thrust::device_vector<float> curr_state {state_dims, 2};
    
    thrust::counting_iterator<size_t> indices {0};
    PopulateCost populate_cost {brownians, sampled_action_trajectories, cost_to_gos, action_trajectory_base, curr_state};

    thrust::for_each(indices, indices + num_samples, populate_cost);
    

    std::cout << "Sampled Trajectories: " << std::endl;
    print_tensor_3D(sampled_action_trajectories, perturbs_dims);
    std::cout << "----------\n" << std::endl;

    std::cout << "Costs to go: " << std::endl;
    print_tensor_3D(cost_to_gos, dim3 {num_samples, 1, num_timesteps});
    std::cout << "----------\n" << std::endl;

}
