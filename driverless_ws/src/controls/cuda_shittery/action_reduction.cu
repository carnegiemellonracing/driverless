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

struct ActionWeightTuple {
    float action;
    float weight;
}

struct ReduceAction {
    ActionWeightTuple *action_weight_t0;
    ActionWeightTuple *action_weight_t1;
    ActionWeightTuple *res;

    ReduceAction() {}

    __device__ ActionWeightTuple *operator()(const ActionWeightTuple *action_weight_t0,
                                             const ActionWeightTuple *action_weight_t1)
        const
    {
        const float w0 = action_weight_t0->weight;
        const float w1 = action_weight_t1->weight;

        res->action = (w0 * action_weight_t0->action +
                       w1 * action_weight_t1->action) /
                      (w0 + w1);
        res->weight = w0 + w1;

        return res;
    }

}

int main() {
    thrust::device_vector<ActionWeightTuple *> vec(2, 0);
    ActionWeightTuple action_weight_t0{};
    action_weight_t0.action = 1;
    action_weight_t0.weight = 1;

    ActionWeightTuple action_weight_t1{};
    action_weight_t0.action = 2;
    action_weight_t0.weight = 2;

    vec[0] = &action_weight_t0;
    vec[1] = &action_weight_t1;

    ReduceAction reduce_action{};

    ActionWeightTuple *res = thrust::reduce(vec.begin(), vec.end(), 0, reduce_action);

    return 0;
}