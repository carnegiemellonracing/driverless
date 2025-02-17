#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <torch/torch.h>
#include <boost/circular_buffer.hpp>
#include <vector>


using namespace torch;
using Experience = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;   //this is how we are going to save expirience in buffer

/*learning hyper-params*/
const int64_t actor_state = 4;//4 car params + cone x/y 
const int64_t actor_layer_1 = 16;
const int64_t actor_layer_2 = 32;
//const int64_t actor_layer_3 = 64;
//const int64_t actor_layer_4 = 32;
const int64_t actor_out = 2;//output size of actor

// input to critic = actor_state , actor_out as critic is for action-value
const int64_t critic_layer_1 = 16;
const int64_t critic_layer_2 = 32;
const int64_t critic_layer_3 = actor_out;
const int64_t critic_out = 1;// output_size of critic
// note: architecture and activation f's of networks can be changed in "model.hpp"

const float actor_lr = 1e-3;      //learning rate for actor = 0.0001
const float critic_lr = 1e-3;     //learning rate of critic= 0.001
const float tau = 0.005;            //rate at which target nets are updated from local nets
const int batch_size = 16;        
const double action_scale = 1.0;     // we are going to scale the output of action by this quantity
const float gamma2 = 0.99;         //forget/discout rate


/*              Critic definition               */
struct CriticImpl: nn::Module
{
    CriticImpl(int state_size, int action_size, int layer1,int layer2, int layer3, int out):
    //CLayer_state(nn::Linear(state_size,state_size)),
    //CLayer_action(nn::Linear(action_size,action_size)),
    CLayer1(nn::Linear(state_size+action_size,layer1)),
    //Cbatch_norm1(layer1),
    CLayer2(nn::Linear(layer1,layer2)),
    //Cbatch_norm2(layer2),
    CLayer3(nn::Linear(layer2,layer3)),
    //Cbatch_norm3(layer3),
    CLayer4(nn::Linear(layer3,out))
    {
        //register_module("CLayer_state",CLayer_state);
        //register_module("CLayer_action",CLayer_action);
        register_module("CLayer1",CLayer1);
        register_module("CLayer2",CLayer2);
        register_module("CLayer3",CLayer3);
        register_module("CLayer4",CLayer4);
        //register_module("Cbatch_norm1",Cbatch_norm1);
        //register_module("Cbatch_norm2",Cbatch_norm2);
        //register_module("Cbatch_norm3",Cbatch_norm3);
        reset_parameters();
    }

	std::pair<double,double> hidden_init(torch::nn::Linear& layer) {
		double lim = 1. / sqrt(layer->weight.sizes()[0]);
		return std::make_pair(-lim, lim);
	}

	void reset_parameters()
	{
		torch::NoGradGuard no_grad;
		auto CLayer1_init = hidden_init(CLayer1);
		torch::nn::init::uniform_(CLayer1->weight, CLayer1_init.first, CLayer1_init.second);
		auto CLayer2_init = hidden_init(CLayer2);
		torch::nn::init::uniform_(CLayer2->weight, CLayer2_init.first, CLayer2_init.second);
		auto CLayer3_init = hidden_init(CLayer3);
		torch::nn::init::uniform_(CLayer3->weight, CLayer3_init.first, CLayer3_init.second);
		torch::nn::init::uniform_(CLayer4->weight, -3e-3, 3e-3);
	}

    torch::Tensor forward(torch::Tensor state , torch::Tensor action) 
    {
        if (state.dim() == 1)
            state = torch::unsqueeze(state, 0);//Returns a new tensor with a dimension of size one inserted at the specified position.

        if (action.dim() == 1)
            action = torch::unsqueeze(action,0);
		//std::cout << action<< std::endl;
            
        //state = torch::tanh(CLayer_state(state));
        //action = torch::tanh(CLayer_action(action));
        
        auto xa = torch::cat({state,action}, /*dim=*/1);
        xa=torch::relu(CLayer1(xa));
        xa=torch::relu(CLayer2(xa));
        xa=torch::tanh(CLayer3(xa));
        xa=CLayer4(xa);
        return xa;
    }
    void print_params()
    {
        for (const auto& p : this->parameters()) 
        {
            std::cout << p << std::endl;
        }
    }

    torch::nn::Linear /*CLayer_state,CLayer_action,*/CLayer1,CLayer2,CLayer3,CLayer4;
    //torch::nn::BatchNorm1d Cbatch_norm1, Cbatch_norm2, Cbatch_norm3;
};
TORCH_MODULE(Critic);       //refer to https://pytorch.org/tutorials/advanced/cpp_frontend.html

/*              Actor definition            */
struct ActorImpl: nn::Module
{
    ActorImpl(int state_size, int layer1,int layer2/*, int layer3, int layer4*/, int out):
    //ALayer1(nn::Linear(nn::LinearOptions(state_size, layer1).bias(false))),
    ALayer1(nn::Linear(state_size,layer1)),
    ALayer2(nn::Linear(layer1,layer2)),
    //ALayer3(nn::Linear(layer2,layer3)),
    //ALayer4(nn::Linear(layer3,layer4)),
    ALayer5(nn::Linear(layer2,out))
    {
        register_module("ALayer1",ALayer1);
        register_module("ALayer2",ALayer2);
        //register_module("ALayer3",ALayer3);
        //register_module("ALayer4",ALayer4);
        register_module("ALayer5",ALayer5);
        reset_parameters();
    }

	std::pair<double,double> hidden_init(torch::nn::Linear& layer) {
		double lim = 1. / sqrt(layer->weight.sizes()[0]);
		return std::make_pair(-lim, lim);
	}

	void reset_parameters()
	{
		torch::NoGradGuard no_grad;
		auto ALayer1_init = hidden_init(ALayer1);
		torch::nn::init::uniform_(ALayer1->weight, ALayer1_init.first, ALayer1_init.second);
		auto ALayer2_init = hidden_init(ALayer2);
		torch::nn::init::uniform_(ALayer2->weight, ALayer2_init.first, ALayer2_init.second);
		torch::nn::init::uniform_(ALayer5->weight, -3e-3, 3e-3);
	}

    torch::Tensor forward(torch::Tensor x, double action_scale) 
    {
       //std::cout << x << std::endl;

        x=torch::relu(ALayer1(x));
        x=torch::relu(ALayer2(x));
        //x=torch::tanh(ALayer3(x));
        //x=torch::tanh(ALayer4(x));
        x=/*action_scale*/torch::tanh(ALayer5(x));
        return x;
    }
    void print_params()
    {
        for (const auto& p : this->parameters()) 
        {
            std::cout << p << std::endl;
        }
    }
    torch::nn::Linear ALayer1,ALayer2/*,ALayer3,ALayer4*/,ALayer5;
};
TORCH_MODULE(Actor);

class ReplayBuffer 
{
        public:
            ReplayBuffer() {}

            void addExperienceState(torch::Tensor state, torch::Tensor action, torch::Tensor reward, torch::Tensor next_state, torch::Tensor done)// add the 5 info to buffer
            {
                addExperienceState(std::make_tuple(state, action, reward, next_state, done));//but first convert them to a tuple
            }

            void addExperienceState(Experience experience) {
             circular_buffer.push_back(experience);//finally add them
            }

         Experience sample(int num_agent)//make a batch of size = num_Agent(rows) x 5(colums)// we need to transpose for batches
         {
			 
			 	std::random_device rd;
				std::mt19937 re(rd());
             Experience experiences;
             Experience exp = sample(std::uniform_int_distribution<uint>(0, num_agent-1)(re));
             experiences = exp;
                
                for (short i = 0; i < num_agent-1; i++) {
                    exp=sample(std::uniform_int_distribution<uint>(0, num_agent-1)(re));
                    std::get<0>(experiences) = torch::cat({std::get<0>(experiences),std::get<0>(exp)}, /*dim=*/0);
                    std::get<1>(experiences) = torch::cat({std::get<1>(experiences),std::get<1>(exp)}, /*dim=*/0);
                    std::get<2>(experiences) = torch::cat({std::get<2>(experiences),std::get<2>(exp)}, /*dim=*/0);
                    std::get<3>(experiences) = torch::cat({std::get<3>(experiences),std::get<3>(exp)}, /*dim=*/0);
                    std::get<4>(experiences) = torch::cat({std::get<4>(experiences),std::get<4>(exp)}, /*dim=*/0);

                 }
             return experiences;
          }

          Experience sample(uint idx) {
                  Experience exp = circular_buffer.at(idx);//sample a data point randomly
                   return exp;
          }

          size_t getLength() {
              return circular_buffer.size();//size of the buffer
          }

          boost::circular_buffer<Experience> circular_buffer{batch_size};//max size of buffer = 10000
};      //refer to https://github.com/EmmiOcean/DDPG_LibTorch/blob/master/replayBuffer.h

class OUNoise {
//"""Ornstein-Uhlenbeck process."""
    private:
        size_t size;
        std::vector<double> mu;
        std::vector<double> state;
        double theta=0.15;
        double sigma=0.1;

    public:
        OUNoise (size_t size_in) {
            size = size_in;
            mu = std::vector<double>(size, 0);
            reset();
        }

        void reset() {
            state = mu;
        }

        std::vector<double> sample(std::vector<double> action) {
        //"""Update internal state and return it as a noise sample."""
            for (size_t i = 0; i < state.size(); i++) {
                auto random = ((double) rand() / (RAND_MAX));
                double dx = theta * (mu[i] - state[i]) + sigma * random;
                state[i] = state[i] + dx;
                action[i] += state[i];
            }
            return action;
    }
};