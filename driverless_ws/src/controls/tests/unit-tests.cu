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

        TEST(MPPITest, ReduceTimestepTest) {
            float *action_trajectories = calloc(num_action_trajectories, sizeof(float));
            float *cost_to_gos = calloc(num_timesteps * num_samples, sizeof(float));
            DeviceAction *averaged_actions = calloc(num_timesteps, sizeof(DeviceAction));
            timestep = 3; // 4th column
            for (size_t i = 0; i < num_samples; i++) {
                float i_float = float(i)
                *(IDX_3D(action_trajectories, action_trajectories_dims, dim3(i, timestep, 0))) = i_float;
                *(IDX_3D(action_trajectories, action_trajectories_dims, dim3(i, timestep, 1))) = i_float * 1.5f
                *(IDX_3D(action_trajectories, action_trajectories_dims, dim3(i, timestep, 2))) = i_float + 10.0f
                // should still be a unweighted average
                cost_to_gos[i * num_timesteps + timestep] = 2;
            }
            ReduceTimestep functor {averaged_actions, action_trajectories, cost_to_gos};
            functor(timestep);
            DeviceAction intended_action = {{511.5f, 767.25f, 512.5f}};
            for (size_t i = 0; i < action_dims; i++) {
                EXPECT_FLOAT_EQ(averaged_actions[timestep].data[i], intended.action.data[i]);
            }
            free(averaged_actions);
            free(action_trajectories);
            free(cost_to_gos);
        }

        TEST(MPPITest, ReduceTimestepTest_hard) {
            float *action_trajectories = calloc(num_action_trajectories, sizeof(float));
            float *cost_to_gos = calloc(num_timesteps * num_samples, sizeof(float));
            DeviceAction *averaged_actions = calloc(num_timesteps, sizeof(DeviceAction));
            timestep = 3; // 4th column
            for (size_t i = 0; i < num_samples; i++) {
                float i_float = float(i)
                *(IDX_3D(action_trajectories, action_trajectories_dims, dim3(i, timestep, 0))) = i_float;
                *(IDX_3D(action_trajectories, action_trajectories_dims, dim3(i, timestep, 1))) = i_float * 1.5f
                *(IDX_3D(action_trajectories, action_trajectories_dims, dim3(i, timestep, 2))) = i_float + 10.0f
                cost_to_gos[i * num_timesteps + timestep] = -std::log(i + 1);
            }
            ReduceTimestep functor {averaged_actions, action_trajectories, cost_to_gos};
            functor(timestep);
            DeviceAction intended_action = {{2050.0f / 3.0f, 1025.0f, 3459225.0f / 43648.0f}};
            for (size_t i = 0; i < action_dims; i++) {
                EXPECT_FLOAT_EQ(averaged_actions[timestep].data[i], intended.action.data[i]);
            }
            free(averaged_actions);
            free(action_trajectories);
            free(cost_to_gos);
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