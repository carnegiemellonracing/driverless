#include <iostream>
#include <cmath>

constexpr float ACTION_DIMS = 3;
constexpr float STATE_DIMS = 10;
constexpr float GRAVITY = 9.81; //in m/s^2
constexpr float CG_TO_FRONT = 1; //in meters TO IMPLEMENT
constexpr float CG_TO_REAR = 1; //in meters TO IMPLEMENT
constexpr float WHEEL_RADIUS = .2; //in meters to IMPLEMENT
constexpr float CAR_MASS = 150; //in KG to IMPLEMENT
constexpr float AERO_CONSTANT = 150; //in Ns^2/m^2 to IMPLEMENT
constexpr float COEF_FRICTION = .1; //unitlss to IMPLEMENT
constexpr float WHEEL_ROTATIONAL_INTERIA = 130; //in kg*m^2 to IMPLEMENT TAKEN ALONG AXLE AXIS
constexpr float CAR_ROTATIONAL_INTERTIA = 130; //in kg*m^2 to IMPLEMENT. TAKEN ALONG Z AXIS THROUGH CG
constexpr float NOSE_CONE_CONSTANT = .3; //in dimensionless to IMPLEMENT. how much of drag becomes downforce



float cosd (float degrees);
float sind(float degrees);
void tireModel (float slip_ratio, float slip_angle, float load, float forces[]); //TO IMPLEMENT
float calculate_slip_ratio(float wheel_speed, float velocity);

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

    tireModel(front_slip_ratio,steering_angle,front_load,front_froces);
    tireModel(rear_slip_ratio,0,rear_load,rear_froces);

    float front_force_x = front_froces[0];
    float front_force_y = front_froces[1];
    float rear_force_x = rear_froces[0];
    float rear_force_y = rear_froces[1];

    //gets drag
    float drag = AERO_CONSTANT*powf(velocity,2);

    //Upadates dot array
    state_dot[0] = x_dot_car* cosd(yaw_world) + y_dot_car*sind(yaw_world);
    state_dot[1] = -x_dot_car* sind(yaw_world) + y_dot_car*cosd(yaw_world);
    state_dot[2] = yaw_rate;
    state_dot[3] = (front_force_x+ rear_force_x-drag)/CAR_MASS-x_dot_car*yaw_rate;    
    state_dot[4] = (front_force_y + rear_force_y)/CAR_MASS + +x_dot_car*yaw_rate;  //assumes that no drag in this direction
    state_dot[5] = (CG_TO_FRONT*front_force_x - CG_TO_REAR*rear_force_x)/CAR_ROTATIONAL_INTERTIA;
    state_dot[6] = 0; //might need to change
    state_dot[7] = drag*NOSE_CONE_CONSTANT; 
    state_dot[8] = (torque_front - COEF_FRICTION*WHEEL_RADIUS*front_load)/WHEEL_ROTATIONAL_INTERIA;
    state_dot[9] = (torque_rear - COEF_FRICTION*WHEEL_RADIUS*rear_load)/WHEEL_ROTATIONAL_INTERIA;
}

//trig functions in degrees
float sind(float degrees){
    return std::sin(degrees*M_PI/180);
}

//trig functions in degrees
float cosd(double degrees){
    return std::cos(degrees*M_PI/180);
}


constexpr float MODEL_SLOPE= 1;
constexpr float MODEL_PEAK = 15;
constexpr float MODEL_FALLOFF = 50;
//calculates slip ratio
//Forces array implanted with X and Y force of wheel
void tireModel(float slip_ratio, float slip_angle, float load, float forces[]){ 
    forces[0] = 8; //X force
    forces [1] = 20; // Y forces
}

//calculates slip ratio
float calculate_slip_ratio(float wheel_speed, float velocity){
    float tangential_velo = wheel_speed*WHEEL_RADIUS;
    return (tangential_velo -velocity)/velocity;

}




int main() {
    std::cout << "Hello world!";

    return 0;
}

