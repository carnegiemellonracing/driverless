#include <gtest/gtest.h>

namespace controls {
    namespace mppi {
        class MPPITest : public testing::Test {
        protected:
            void SetUp() override {
                device_action1 {{1.0F, 2.0F, 3.0F}};
                device_action2 {{0.2F, 3.0F, 1.5F}};
                device_action3 {{-0.6F, 1.4F, 1.0F}};
            }

            DeviceAction device_action1;
            DeviceAction device_action2;
            DeviceAction device_action3;

        };
    }


    TEST(controls_lib, stub_test) {
        ASSERT_EQ(1, 1);

    }
}

class ModelTest : public testing::Test {
protected:

    ModelTest() {
        // You can do set-up work for each test here.
    }

    // Class members declared here can be used by all tests in the test suie
};

    TEST_F(ModelTest, sanityChecks){

}


// namespace model {
//     bool arrequals(float[] arr1, float[]arr2, int length)
//     {
//         for (int i=0; i<length; i++) {
//              if(arr1[i] != arr2[i]){
//                 return false;
//              }
//         }
//
//         return true;
//     }
//
//     void copy(float[] arr1, float[]arr2, int length)
//     {
//         for (int i=0; i<length; i++) {
//              arr2[i] = arr1[i];
//         }
//     }


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
    float action[3] = [0,0,0];
    dynamics(state, action, state_dot_res);
    EXPECT_EQ(equals(state_dot_ex,state_dot_res), true)
}


TEST(ModelTest, Nothing_Gives_Nothing) {
    //rand stuff
    float state[10] = [0,0,0,0,0,0,0,0,0,0];
    float state_dot_ex[10] = [0,0,0,0,0,0,0,0,0,0];
    float state_dot_res[10] = [0,0,0,0,0,0,0,0,0,0];
    float action[3] = [0,0,0];
    dynamics(state, action, state_dot_res);
    EXPECT_EQ(equals(state_dot_ex,state_dot_res), true)
}

TEST(ModelTest, LEFT_GIVES_LEFT) {
    //rand stuff
    float state[10] = [0,0,0,0,0,0,0,0,0,0];
    float state_dot_res[10] = [0,0,0,0,0,0,0,0,0,0];
    float action[3] = [-15,45,45];
    dynamics(state, action, state_dot_res);
    EXPECT_EQ(state_dot_res[0]>0,true)
    EXPECT_EQ(state_dot_res[1]<0,true)
    EXPECT_EQ(state_dot_res[3]>0,true)
    EXPECT_EQ(state_dot_res[4]<0,true)
    EXPECT_EQ(state_dot_res[5]<0,true)
}

TEST(ModelTest, RIGHT_GIVES_RIGHT) {
    //rand stuff
    float state[10] = [0,0,0,0,0,0,0,0,0,0];
    float state_dot_res[10] = [0,0,0,0,0,0,0,0,0,0];
    float action[3] = [15,45,45];
    dynamics(state, action, state_dot_res);
    EXPECT_EQ(state_dot_res[0]>0,true)
    EXPECT_EQ(state_dot_res[1]>0,true)
    EXPECT_EQ(state_dot_res[3]>0,true)
    EXPECT_EQ(state_dot_res[4]>0,true)
    EXPECT_EQ(state_dot_res[5]>0,true)

}


TEST(ModelTest, RIGHT_GIVES_RIGHT) {
    //rand stuff
    float state[10] = [0,0,0,0,0,0,0,0,0,0];
    float state_dot_res[10] = [0,0,0,0,0,0,0,0,0,0];
    float action[3] = [15,45,45];
    dynamics(state, action, state_dot_res);
    EXPECT_EQ(state_dot_res[0]>0,true)
    EXPECT_EQ(state_dot_res[1]>0,true)
    EXPECT_EQ(state_dot_res[3]>0,true)
    EXPECT_EQ(state_dot_res[4]>0,true)
    EXPECT_EQ(state_dot_res[5]>0,true)

}


int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
