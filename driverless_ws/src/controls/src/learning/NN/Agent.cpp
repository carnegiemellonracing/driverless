#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <torch/torch.h>
#include <boost/circular_buffer.hpp>
#include <vector>

#include "Agent.hpp"

using namespace std;
using namespace torch;
using Experience = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;


Agent::Agent():
a_local(Actor(actor_state, actor_layer_1, actor_layer_2/*, actor_layer_3, actor_layer_4*/, actor_out)),
a_target(Actor(actor_state, actor_layer_1, actor_layer_2/*, actor_layer_3, actor_layer_4*/, actor_out)),
critic_1(Critic(actor_state, actor_out, critic_layer_1, critic_layer_2, critic_layer_3, critic_out)),
critic_2(Critic(actor_state, actor_out, critic_layer_1, critic_layer_2, critic_layer_3, critic_out)),
critic_1_target(Critic(actor_state, actor_out, critic_layer_1, critic_layer_2, critic_layer_3, critic_out)),
critic_2_target(Critic(actor_state, actor_out, critic_layer_1, critic_layer_2, critic_layer_3, critic_out)),
actor_optimizer(a_local->parameters(), actor_lr),
critic_optimizer_1(critic_1->parameters(), critic_lr),
critic_optimizer_2(critic_2->parameters(), critic_lr),
Buffer(ReplayBuffer()),
noise(OUNoise(actor_out)),
device(torch::kCPU)
{
    Agent::id = id;
    if (torch::cuda::is_available()) 
    {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    a_local->to(device);
    a_target->to(device);
    critic_1->to(device);
    critic_2->to(device);
    critic_1_target->to(device);
    critic_2_target->to(device);
   
    a_local->to(torch::kF64);
    a_target->to(torch::kF64);
    critic_1->to(torch::kF64);
    critic_2->to(torch::kF64);
    critic_1_target->to(torch::kF64);
    critic_2_target->to(torch::kF64);
     
    Agent::hard_copy();
}

//NEEDS TO CHANGE
float Agent::get_reward(float &reward, float (&reward_type)[5], vector<float> &new_state, vector<float> &old_state)
{
    //target is to reach the top cos(pi) = -1
    float new_distance = abs((-1) - new_state[1]);
    float old_distance = abs((-1) - old_state[1]);
    reward = 100*(old_distance-new_distance);
    reward_type[0] += reward; // task reward

    reward -= abs(new_state[3]);
    reward_type[1] -= abs(new_state[3]);// negative reward for more torque

    reward -= 0.1*new_state[4];
    reward_type[2] -= 0.1*new_state[4];// negative reward for more time taken

    if(new_distance<0.02 && abs(new_state[2])<0.3)// if we are near the top and our speed is less
    {
        reward += 1000;
        reward_type[3] += 1000;// task successful
        return 1.0;
    }
    else if(abs(new_state[1])>10)//else if our speed is too much
    {
        reward -=1000;
        reward_type[4] -= 1000;// safety breached shut down
        return 0.0;
    }
    //else
    //{
        return 0.0;
    //}

}
 //TO CHANGE
void Agent::step(TestEnvironment &env, double &done, double &reward, double &total_reward)
{
    //a_local->eval();
    //NoGradGuard gaurd;

    //torch::NoGradGuard no_grad;
    
    torch::Tensor current_state = env.State();

	torch::Tensor mu = torch::full({1,actor_out}, 0.);// = torch::zeros({1, 1}, torch::kF64);
	torch::Tensor std = torch::full(1, 0.8);// = torch::zeros({1, 1}, torch::kF64);
	   

    auto action_tensor = a_local->forward(current_state , action_scale);//+ 
				//torch::Tensor(at::normal(mu,(std).expand_as(mu)));//.to(torch::kCPU);//global action scale
	
	{
	torch::NoGradGuard no_grad;
	action_tensor = torch::clamp(action_tensor+torch::Tensor(at::normal(mu,(std).expand_as(mu))) ,-1*action_scale,action_scale);//.to(torch::kCPU);//global action scale
	}

    //vector<double> action_vector(action_tensor_cpu.data_ptr<float>() , action_tensor_cpu.data_ptr<float>() + action_tensor_cpu.numel());
    //action_vector = Agent::noise.sample(action_vector);   //adding noise to the action vector

    double x_act = action_tensor[0][0].item<double>();
    double y_act = action_tensor[0][1].item<double>();
    auto sd = env.Act(x_act, y_act);

    torch::Tensor new_state = env.State();

    torch::Tensor reward_tensor = env.Reward(std::get<1>(sd));
    torch::Tensor done_tensor = std::get<2>(sd);
    
    reward=reward_tensor[0][0].item<double>();
    total_reward += reward_tensor[0][0].item<double>();
    done= done_tensor[0][0].item<double>();

    Agent::Buffer.addExperienceState(current_state, action_tensor, reward_tensor, new_state, done_tensor);
}


void Agent::learn(double &total_actor_loss, double &total_critic_loss, int &counter)
{
	//cout << "Buffer length:" << Agent::Buffer.getLength() << "counter:" << counter << endl;
	
    if((counter)%batch_size== 0 && counter!=0)//batchsixe global
    {
		//cout << "we are in" << endl;
        //a_local->train();
        //auto& [state, action, reward_batch, next_state, donezo] = Buffer.sample(batch_size);//global batch_size

        auto Experiences=Buffer.sample(batch_size);
        
		auto state = std::get<0>(Experiences);
		auto action = std::get<1>(Experiences);
		auto reward_batch = std::get<2>(Experiences);
		auto next_state = std::get<3>(Experiences);
		auto donezo = std::get<4>(Experiences);        

		torch::Tensor mu = torch::full({batch_size,actor_out}, 0.);// = torch::zeros({1, 1}, torch::kF64);
		torch::Tensor std = torch::full(1, 0.2);// = torch::zeros({1, 1}, torch::kF64);

		//std::cout << "n_s" << endl;
		//std::cout << next_state << std::endl;    
		

		
		auto actions_next = a_target->forward(next_state,action_scale);
		
		//std::cout << "a_nxt0" << endl;
		//std::cout << actions_next << std::endl;		
		{
		torch::NoGradGuard no_grad;
		actions_next =	actions_next+ torch::clamp(torch::Tensor(at::normal(mu,(std).expand_as(mu))),-0.5,0.5);

		actions_next = torch::clamp(actions_next,-1*action_scale,action_scale);
		}
		//std::cout << "a_nxt1" << endl;
		//std::cout << actions_next << std::endl;

			
		auto Q_targets_next_1 = critic_1_target->forward(next_state, actions_next);
		auto Q_targets_next_2 = critic_2_target->forward(next_state, actions_next);
		//Q_targets_next_1 = Q_targets_next_1 * (1 - done);
		//Q_targets_next_2 = Q_targets_next_2 * (1 - done);

		auto Q_targets_next = torch::min(Q_targets_next_1, Q_targets_next_2);
		
		auto Q_expected_1 = critic_1->forward(state, action); 
		auto Q_expected_2 = critic_2->forward(state, action);

		//std::cout << "reward" << endl;
		//std::cout << reward_batch << std::endl;
		//std::cout << "Qt_n" << endl;
		//std::cout << Q_targets_next << std::endl;
		auto Q_targets = reward_batch + (gamma2 * Q_targets_next * (1 - donezo));
		//std::cout << "Qt" << endl;
		//std::cout << Q_targets << std::endl;

		//phil put critic into training mode
		torch::Tensor critic_loss_1 = torch::mse_loss(Q_expected_1, Q_targets.detach());
		torch::Tensor critic_loss_2 = torch::mse_loss(Q_expected_2, Q_targets.detach());
		torch::Tensor critic_loss = critic_loss_1 + critic_loss_2;
		critic_optimizer_1.zero_grad();
		critic_optimizer_2.zero_grad();
		critic_loss.backward();

		critic_optimizer_1.step();
		critic_optimizer_2.step();
        
        int learner_count=counter/batch_size;
        //cout << "learner_count" << learner_count<< endl;
		if(learner_count%2!=0)
		{return;}
		
		auto actions_pred = a_local->forward(state, action_scale);
		auto actor_loss = -(critic_1->forward(state, actions_pred).mean());

		actor_optimizer.zero_grad();
		actor_loss.backward();
		actor_optimizer.step();

// ----------------------- update target networks ----------------------- #

        Agent::soft_copy();
        total_actor_loss += actor_loss.item<double>();
        total_critic_loss += critic_loss.item<double>();
    }
    //Agent::Buffer.empty();
}

void Agent::hard_copy()
{
    torch::NoGradGuard no_grad;
    for(size_t i=0; i < a_target->parameters().size(); i++)
        a_target->parameters()[i].copy_(a_local->parameters()[i]);
    for(size_t i=0; i < critic_1_target->parameters().size(); i++)
        critic_1_target->parameters()[i].copy_(critic_1->parameters()[i]);
    for(size_t i=0; i < critic_2_target->parameters().size(); i++)
        critic_2_target->parameters()[i].copy_(critic_2->parameters()[i]);

}

void Agent::soft_copy()
{
    torch::NoGradGuard no_grad;//       disables calulation of gradients
    for(size_t i=0; i < a_target->parameters().size(); i++)
        a_target->parameters()[i].copy_(tau * a_local->parameters()[i] + (1.0 - tau) * a_target->parameters()[i]);//global tau
    for(size_t i=0; i < critic_1_target->parameters().size(); i++)
        critic_1_target->parameters()[i].copy_(tau * critic_1->parameters()[i] + (1.0 - tau) * critic_1_target->parameters()[i]);//global tau
    for(size_t i=0; i < critic_2_target->parameters().size(); i++)
        critic_2_target->parameters()[i].copy_(tau * critic_2->parameters()[i] + (1.0 - tau) * critic_2_target->parameters()[i]);//global tau

}
