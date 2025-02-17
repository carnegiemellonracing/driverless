#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <torch/torch.h>
#include <boost/circular_buffer.hpp>
#include <vector>
#include <random>

#include "NN/Agent.hpp"
//#include "TestEnvironment.h"

using namespace std;

int main(int argc, const char** argv)
{
	
    // Random engine.
    std::random_device rd;
    std::mt19937 re(rd());
    std::uniform_int_distribution<> dist(-5, 5);//Produces random 
    //integer values i, uniformly distributed on the closed interval [-5, 5]

    // Environment.
    double x = double(dist(re)); // goal x pos
    double y = double(dist(re)); // goal y pos
    TestEnvironment env(x, y); //initlizing the environment with 
    //_pos, goal_ and state_
    Agent agent;
    
    // Output.
    std::ofstream out;
    out.open("../data/data.csv");

    // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
    out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";

	int c=0;
	uint n_iter = 10000;
    int n_epochs = 30;

    for(int epoch=1; epoch<=n_epochs; epoch++)
    {

        for (uint i=0;i<n_iter;i++)
        {
			
			double total_reward = 0.0;
			double reward = 0.0;
			double done = 0.0;
			double total_actor_loss = 0.0;
			double total_critic_loss = 0.0;

            agent.step(env, done, reward, total_reward);
            
            out << epoch << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << done << "\n";

            agent.learn(total_actor_loss, total_critic_loss, c);
            c++;
            
            if(done>0.5)
            {
				cout<< " epoch:" << epoch << endl;
				// Set new goal.
                double x_new = double(dist(re)); 
                double y_new = double(dist(re));
                env.SetGoal(x_new, y_new);

                // Reset the position of the agent.
                env.Reset();

                // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
                out << epoch << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";
			}
        };
        
                // Reset at the end of an epoch.
        double x_new = double(dist(re)); 
        double y_new = double(dist(re));
        env.SetGoal(x_new, y_new);

        // Reset the position of the agent.
        env.Reset();

        // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST, RESETTING)
        out << epoch << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";

                
    }
     out.close();
   
    return 0;
}