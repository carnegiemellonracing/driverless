
#include<torch>

class TD3_Agent {
    float policy_noise; // = .2 * self.max_action
    float noise_clip; // = .5 * self.max_action
    float tau = .005;
    float delay_counter = 0;
    
}