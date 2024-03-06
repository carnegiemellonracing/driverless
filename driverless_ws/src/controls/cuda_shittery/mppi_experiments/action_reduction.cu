#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <cmath>

#include "config.cuh"
#include "cuda_utils.cuh"

struct ActionWeightTuple {
    Action action;
    float weight;
};

struct ActionAverage {
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

    __device__ IndexToActionWeightTuple (const float* action_trajectories, const float* cost_to_gos)
        : action_trajectories {action_trajectories},
          cost_to_gos {cost_to_gos} {}

    __device__ ActionWeightTuple operator() (const size_t idx) const {
        ActionWeightTuple res {};

        const size_t i = idx % num_samples;
        const size_t j = idx / num_samples;
        memcpy(&res.action.data, IDX_3D(action_trajectories, perturbs_dims, dim3(i, j, 0)), sizeof(float) * action_dims);

        const float cost_to_go = cost_to_gos[idx];
        res.weight = __expf(-1.0f / temperature * cost_to_go);

        return res;
    }
};

struct ReduceTimestep {
    Action* averaged_actions;

    const float* action_trajectories;
    const float* cost_to_gos;

    const ActionAverage reduction_functor {};

    ReduceTimestep (Action* averaged_actions, const float* action_trajectories, const float* cost_to_gos)
        : averaged_actions {averaged_actions},
          action_trajectories {action_trajectories},
          cost_to_gos {cost_to_gos} { }

    __device__ void operator() (const size_t idx) {
        thrust::counting_iterator<size_t> indices {idx * num_samples};

        averaged_actions[idx] = thrust::transform_reduce(
            thrust::device,
            indices, indices + num_samples,
            IndexToActionWeightTuple {action_trajectories, cost_to_gos},
            ActionWeightTuple { Action {}, 0 }, reduction_functor).action;
    }
};

int main() {
    thrust::device_vector<float> sampled_action_trajectories { 1, 1, 2, 2, 3, 3, 4, 4}; // (num_perturbs, 4.1);
    thrust::device_vector<float> cost_to_gos {1, 2, 2, 1};// (num_samples * num_timesteps, 1);
    thrust::device_vector<Action> averaged_actions (num_timesteps);

    thrust::counting_iterator<size_t> indices {0};

    for (size_t i = 0; i < 1000000; i++) {
        thrust::for_each(indices, indices + num_timesteps, ReduceTimestep {
            averaged_actions.data().get(),
            sampled_action_trajectories.data().get(),
            cost_to_gos.data().get()
        });
        cudaDeviceSynchronize();

        if (i % 1000 == 0) {
            std::cout << "i: " << i << std::endl;
        }
    }

    // print averaged_action like in print_tensor_3d
    thrust::host_vector<Action> averaged_actions_host = averaged_actions;

    for (int i = 0; i < num_timesteps; i++) {
        std::cout << "{ ";
        for (int j = 0; j < action_dims; j++) {
            std::cout << averaged_actions_host[i].data[j] << " ";
        }
        std::cout << "}\n";
    }

    return 0;
}