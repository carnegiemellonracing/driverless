#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>

#include <iostream>
#include <cmath>

#include "config.cuh"
#include "cuda_utils.cuh"

struct ActionWeightTuple {
    Action action;
    float weight;
};

struct ReduceAction {
    __device__ ActionWeightTuple operator()(const ActionWeightTuple& action_weight_t0,
                                            const ActionWeightTuple& action_weight_t1) const
    {
        ActionWeightTuple res {};

        const float w0 = action_weight_t0.weight;
        const float w1 = action_weight_t1.weight;

        res.action = (w0 * action_weight_t0.action + w1 * action_weight_t1.action) / (w0 + w1);
        res.weight = w0 + w1;

        return res;
    }
};

struct IndexToActionWeightTuple {
    const float* action_trajectories;
    const float* cost_to_gos;

    IndexToActionWeightTuple (const float* action_trajectories, const float* cost_to_gos)
        : action_trajectories {action_trajectories},
          cost_to_gos {cost_to_gos} {}

    __device__ ActionWeightTuple operator() (const size_t idx) const {
        ActionWeightTuple res {};

        const size_t j = idx % num_timesteps;
        const size_t i = idx / num_timesteps;
        memcpy(&res.action.data, IDX_3D(action_trajectories, perturbs_dims, dim3(i, j, 0)), sizeof(float) * action_dims);

        const float cost_to_go = cost_to_gos[idx];
        res.weight = __expf(-1.0f / temperature * cost_to_go);

        return res;
    }
};

int main() {
    thrust::device_vector<float> sampled_action_trajectories (num_perturbs, 4.1);
    thrust::device_vector<float> cost_to_gos (num_samples * num_timesteps, 1);

    thrust::counting_iterator<size_t> indices {0};

    Action action = thrust::transform_reduce(
        indices, indices + num_samples * num_timesteps,
        IndexToActionWeightTuple {sampled_action_trajectories.data().get(), cost_to_gos.data().get()},
        ActionWeightTuple { Action {}, 0 }, ReduceAction {}).action;

    std::cout << "Action: ";
    for (size_t i = 0; i < action_dims; i++) {
        std::cout << action.data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}