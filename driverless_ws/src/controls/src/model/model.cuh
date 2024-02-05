#include <iostream>
#include <cmath>

//SOURSE: SEE OVERLEAF MPPI DOCUMENTATION

//model constants
constexpr float ACTION_DIMS = 3;
constexpr float STATE_DIMS = 10;

//physics constants
constexpr float GRAVITY = 9.81; //in m/s^2
constexpr float CG_TO_FRONT = 2; //in meters TO IMPLEMENT
constexpr float CG_TO_REAR = 1; //in meters TO IMPLEMENT
constexpr float WHEEL_RADIUS = .2; //in meters to IMPLEMENT
constexpr float CAR_MASS = 150; //in KG to IMPLEMENT
constexpr float LOGITUDNAL_AERO_CONSTANT = 150; //in Ns^2/m^2 to IMPLEMENT
constexpr float LATERAL_AERO_CONSTANT = 150; //in Ns^2/m^2 to IMPLEMENT
constexpr float COEF_FRICTION = .1; //unitlss to IMPLEMENT
constexpr float WHEEL_ROTATIONAL_INTERIA = 130; //in kg*m^2 to IMPLEMENT TAKEN ALONG AXLE AXIS
constexpr float CAR_ROTATIONAL_INTERTIA = 130; //in kg*m^2 to IMPLEMENT. TAKEN ALONG Z AXIS THROUGH CG
constexpr float NOSE_CONE_CONSTANT = .3; //in dimensionless to IMPLEMENT. how much of drag becomes downforce


//tire model constants
constexpr float max_force_x = 43; //Maximum force x TO IMPLEMENT
constexpr float slip_ratio_max_x= 1; //slip ratio that yields the max force TO IMPLEMENT
constexpr float post_saturation_force_x= 1; // After tires start slipping what force we get
constexpr float max_force_y = 10; //Maximum force Y TO IMPLEMENT
constexpr float slip_ratio_max_y= 1; //slip ratio that yields the max force TO IMPLEMENT
constexpr float post_saturation_force_y= 1; // After tires start slipping what force we get

//trig functions in degrees
inline float cosd(double degrees){
    return std::cos(degrees*M_PI/180);
}

inline float cotd(float degrees);
inline float tand(float degrees);
inline float sind(float degrees);
void tireModel (float slip_ratio, float slip_angle, float load, float forces[]); //TO IMPLEMENT
inline float calculate_slip_ratio(float wheel_speed, float velocity);

 /*state (in order):
[0] X_world m
[1] Y_world m 
[2] Yaw_World deg
[3] X_dot_Car m/s
[4] Y_dot_Car m/s
[5] Yaw_Rate deg/s
[6] Pitch Moment Nm
[7] Downforce N 
[8] Front Wheel Speed rad/s
[9] rear Wheel Speed rad/s
*/
void dynamics(const float state[], const float action[], float state_dot[]){
    //unpackages state array
    float x_world = state[0];
    float y_world = state[1];
    float yaw_world = state[2];
    float x_dot_car = state[3];
    float y_dot_car = state[4];
    float yaw_rate = state[5];
    float pitch_moment = state[6];
    float downforce = state[7];
    float front_wheel_speed = state[8];
    float rear_wheel_speed = state[9];

    //unpackages action
    float steering_angle = action[0];
    float torque_front = action[1];
    float torque_rear = action[2];

    float velocity = sqrtf(powf(x_dot_car,2) + powf(y_dot_car,2));


    //compures wheel forces
    float front_froces[2];
    float rear_froces[2];
    float front_slip_ratio = calculate_slip_ratio(front_wheel_speed,velocity);
    float rear_slip_ratio = calculate_slip_ratio(rear_wheel_speed,velocity);

    float front_load = CAR_MASS + downforce + pitch_moment/CG_TO_FRONT;
    float rear_load = CAR_MASS + downforce - pitch_moment/CG_TO_REAR;

    tireModel(front_slip_ratio,steering_angle,front_load,front_froces); // WRONG SINCE SLIP ANGLE NOT EQUAL TO STEERING ANGLE CAUSE AS 
    //TIRE DEFOMS THE DIRECTION OF ITS TRAVEL IS NOT SAME AS STEERING ANGLE
    tireModel(rear_slip_ratio,0,rear_load,rear_froces);

    float front_force_x = front_froces[0];
    float front_force_y = front_froces[1];
    float rear_force_x = rear_froces[0];
    float rear_force_y = rear_froces[1];

    //gets drag
    float drag_x = LOGITUDNAL_AERO_CONSTANT*powf(x_dot_car,2);
    float drag_y = LATERAL_AERO_CONSTANT*powf(y_dot_car,2);


    //Upadates dot array
    state_dot[0] = x_dot_car* cosd(yaw_world) + y_dot_car*sind(yaw_world);
    state_dot[1] = -x_dot_car* sind(yaw_world) + y_dot_car*cosd(yaw_world);
    state_dot[2] = yaw_rate;
    state_dot[3] = (front_force_x+ rear_force_x-drag_x)/CAR_MASS-x_dot_car*yaw_rate;    
    state_dot[4] = (front_force_y + rear_force_y-drag_y)/CAR_MASS + +x_dot_car*yaw_rate;  
    state_dot[5] = (CG_TO_FRONT*front_force_x - CG_TO_REAR*rear_force_x)/CAR_ROTATIONAL_INTERTIA;
    state_dot[6] = 0; //might need to change
    state_dot[7] = 2*drag_x*NOSE_CONE_CONSTANT*x_dot_car*state_dot[3]; 
    state_dot[8] = (torque_front - COEF_FRICTION*WHEEL_RADIUS*front_load)/WHEEL_ROTATIONAL_INTERIA;
    state_dot[9] = (torque_rear - COEF_FRICTION*WHEEL_RADIUS*rear_load)/WHEEL_ROTATIONAL_INTERIA;
}

//trig functions in degrees
inline float sind(float degrees){
    return std::sin(degrees*M_PI/180);
}

//trig functions in degrees
inline float tand(float degrees){
    return std::tan(degrees*M_PI/180);
}

//trig functions in degrees
inline float cotd(float degrees){
    return  1.0f / std::tan(degrees*M_PI/180);
}


//calculates slip ratio
//Forces array implanted with X and Y force of wheel
//see overleaf document in MMPI documentation
void tireModel(float slip_ratio, float slip_angle, float load, float forces[]){ 
    
    //gets x force
    if(slip_ratio<slip_ratio_max_x){
        int numerator = load * slip_ratio * max_force_y;
        int within_sqrt = powf(tand(slip_angle),2) + powf((max_force_y/max_force_x),2);
        int denominator = slip_ratio_max_x * std::sqrtf(within_sqrt);
        forces[0] = numerator / denominator;
    }
    else{
        int numerator = load * post_saturation_force_x * max_force_y;
        int within_sqrt = powf(tand(slip_angle),2) + powf((max_force_y/max_force_x),2);
        int denominator = max_force_x * std::sqrtf(within_sqrt);
        forces[0] = numerator / denominator;
    }

    //computes y force
    if(slip_ratio<slip_ratio_max_y){
        int numerator = load* slip_ratio * max_force_x;
        int within_sqrt = powf(cotd(slip_angle),2) + powf((max_force_x/max_force_y),2);
        int denominator = slip_ratio_max_y * std::sqrtf(within_sqrt);
        forces[1] = numerator / denominator;
    }
    else{
        int numerator = load*post_saturation_force_y * max_force_x;
        int within_sqrt = powf(cotd(slip_angle),2) + powf((max_force_x/max_force_y),2);
        int denominator = max_force_y * std::sqrtf(within_sqrt);
        forces[1] = numerator / denominator;
    }

}

//calculates slip ratio
inline float calculate_slip_ratio(float wheel_speed, float velocity){
    float tangential_velo = wheel_speed*WHEEL_RADIUS;
    return (tangential_velo -velocity)/velocity;

}




int main() {
    float state[] = {1,1,90,3,5,0,0,0,0,0}; 
    float control[] = {0,0,0}; 
    float statedot[] = {0,0,0,0,0,0,0,0,0,0}; 

    dynamics(state,control,statedot);
    for (const auto &element : statedot) {
        std::cout << element << " ";
    }
    return 0;
}

