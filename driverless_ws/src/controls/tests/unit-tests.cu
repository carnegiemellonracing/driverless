#include <gtest/gtest.h>
#include <mppi/functors.cuh>

namespace controls {
    namespace mppi {
        //for Copilot
        action_dims = 3;
        struct DeviceAction {
            float data[action_dims];
        };


        class MPPITest : public testing::Test {
        protected:
            void SetUp() override {
                device_action1 = {{1.0F, 2.0F, 3.0F}};
                device_action2 = {{0.2F, 3.1F, -1.5F}};
                device_action3 = {{1.2F, 5.1F, 1.5F}};
                awt1 = {device_action1, 2.0F};
                awt2 = {device_action2, 3.0F};
            }

            DeviceAction device_action1;
            DeviceAction device_action2;
            DeviceAction device_action3;
            ActionWeightTuple awt1;
            ActionWeightTuple awt2;
            ActionWeightTuple awt3;

            action

        };

        TEST(MPPITest, AddActionsTest_Positive) {
            AddActions functor;
            DeviceAction result = functor(device_action1, device_action2);
            for (size_t i = 0; i < action_dims; i++) {
                EXPECT_FLOAT_EQ(result.data[i], device_action3.data[i]);
            }
        }

        TEST(MPPITest, ActionAverageTest) {
            ActionAverage functor;
            ActionWeightTuple result = functor(awt1, awt2);
            ActionWeightTuple intended = {{{0.52F, 2.66F, 0.3F}}, 5.0F};
            EXPECT_FLOAT_EQ(result.weight, intended.weight);
            for (size_t i = 0; i < action_dims; i++) {
                EXPECT_FLOAT_EQ(result.action.data[i], intended.action.data[i])
            }
        }

        TEST(MPPITest, IndexToActionWeightTupleTest) {
            float *action_trajectories = calloc(num_action_trajectories, sizeof(float));
            float *cost_to_gos = calloc(num_timesteps * num_samples, sizeof(float));
            action_trajectories[142851] = 1.2F;
            action_trajectories[142851] = 5.1F;
            action_trajectories[142851] = 1.5F;
            cost_to_gos[1400] = 3.0F; // becomes (372, 1, 0)
            IndexToActionWeightTuple functor {action_trajectories, cost_to_gos};
            ActionWeightTuple result = functor(1400)
            ActionWeightTuple intended = {device_action3, __expf(-3.0F);}
            EXPECT_FLOAT_EQ(result.weight, intended.weight);
            for (size_t i = 0; i < action_dims; i++) {
                EXPECT_FLOAT_EQ(result.action.data[i], intended.action.data[i])
            }
            free(action_trajectories);
            free(cost_to_gos)
        }


    }

}



int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace  model{
    class MODELTests
}
}