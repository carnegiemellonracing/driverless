#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <torch/torch.h>
#include <vector>

#include "model.hpp"
#include "TestEnvironment.h"

using namespace std;
using namespace torch;



class Agent
{
    public:
        Agent();
        void hard_copy();//to hard copy the params of local and target networks
        void soft_copy();//to soft copy the params of local and target networks
        float get_reward(float &reward, float (&reward_type)[5], vector<float> &new_state, vector<float> &old_state);//calculate the reward
        void step(TestEnvironment &env, double &done, double &reward, double &total_reward);
        void learn(double &total_actor_loss, double &total_critic_loss, int &counter);//sample a batch and learn from it.

        Actor a_local;
        Actor a_target;


        torch::optim::Adam actor_optimizer;

		Critic critic_1;
		Critic critic_2;

		
		Critic critic_1_target;
		Critic critic_2_target;


		torch::optim::Adam critic_optimizer_1;
		torch::optim::Adam critic_optimizer_2;

        torch::Device device;

        ReplayBuffer Buffer;
        OUNoise noise;
        int id;


};