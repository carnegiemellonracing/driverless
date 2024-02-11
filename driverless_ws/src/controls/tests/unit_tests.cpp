#include <gtest/gtest.h>
#include <mppi/functors.cuh>


namespace model {

    bool arrequals(float[] arr1, float[]arr2, int length)
    @requires sizeof(arr1) == length;
    @requires sizeof(arr2) == length;
    {
        for (int i=0; i<length; i++) {
             if(arr1[i] != arr2[i]){
                return false;
             }
        }

        return true;
    }

    void copy(float[] arr1, float[]arr2, int length)
    @requires sizeof(arr1) == length;
    @requires sizeof(arr2) == length;
    {
        for (int i=0; i<length; i++) {
             arr2[i] = arr1[i];
        }
    }


TEST(ModelTest, NoDependenceOnWorldCords) {
    //rand stuff
    float state[10] = [0,0,54,3,3,-3,42,-43,2,89];
    float state_dot[10] = [0,0,0,0,0,0,0,0,0,0];
    float action[10] = [-2,12,-65];
   //Tests that nothing matters based on X_World and Y_world
    dynamics(state, action, state_dot);
    float state_dot_2 = copy(state_dot,state_dot_2, 10);
    state[0] = 12;
    state[1] = -12;
    dynamics(state, action, state_dot);
    EXPECT_EQ(equals(state_dot_2,state_dot), true)
}


TEST(ModelTest, Nothing_Gives_Nothing) {
    //rand stuff
    float state[10] = [0,0,0,0,0,0,0,0,0,0];
    float state_dot_ex[10] = [0,0,0,0,0,0,0,0,0,0];
    float state_dot_res[10] = [0,0,0,0,0,0,0,0,0,0];
    float action[10] = [0,0,0];
    dynamics(state, action, state_dot_res);
    EXPECT_EQ(equals(state_dot_ex,state_dot_res), true)
}


TEST(ModelTest, Nothing_Gives_Nothing) {
    //rand stuff
    float state[10] = [0,0,0,0,0,0,0,0,0,0];
    float state_dot_ex[10] = [0,0,0,0,0,0,0,0,0,0];
    float state_dot_res[10] = [0,0,0,0,0,0,0,0,0,0];
    float action[10] = [0,0,0];
    dynamics(state, action, state_dot_res);
    EXPECT_EQ(equals(state_dot_ex,state_dot_res), true)
}



} //model namespace



int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
