#include <gtest/gtest.h>


namespace model {

    bool equals(float[] arr1, float[]arr2, int length)
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


    // The fixture for testing class Foo.
    class ModelTest : public testing::Test {
    protected:

        ModelTest() {
            // You can do set-up work for each test here.
        }

        float state[10] = [0,0,0,0,0,0,0,0,0,0];
        float state_dot[10] = [0,0,0,0,0,0,0,0,0,0];
        float action[10] = [0,0,0];

    };

TEST_F(ModelTest, sanityCheck) {
   //Tests that nothing matters based on X_World and Y_world
   dynamics(state, action, state_dot);

}

} //model namespace


TEST(controls_lib, stub_test) {
ASSERT_EQ(1, 1);
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
