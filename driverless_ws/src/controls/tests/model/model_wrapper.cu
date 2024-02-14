#include <model/bicycle/model.cuh>
#include <types.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace controls;

    State state;
    Action action;

    for (uint8_t i = 0; i < state_dims; i++) {
        state[i] = strtof(argv[i + 1], nullptr);
    }

    for (uint8_t i = 0; i < action_dims; i++) {
        if (i + state_dims >= argc) {
            throw std::runtime_error("Not enough arguments.");
        }

        action[i] = strtof(argv[state_dims + i + 1], nullptr);
    }

    State statedot;
    model::bicycle::dynamics(state.data(), action.data(), statedot.data());

    for (uint8_t i = 0; i < state_dims; i++) {
        std::cout << statedot[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}