#include <rclcpp/rclcpp.hpp>
#include <constants.hpp>
#include <types.hpp>
#include <interfaces/msg/control_action.hpp>

namespace controls {
    namespace sysid {
        class SysIdNode : public rclcpp::Node
        {
            public:
                SysIdNode(int selected_test);
                void slow_accel_deccel();
                void fast_accel_deccel();
                void full_clockwise_ramp();
                void full_anticlockwise_ramp();
                void half_clockwise_ramp();
                void half_anticlockwise_ramp();
            private:

                void get_next_message();
                void on_twist(const TwistMsg& msg);

                rclcpp::TimerBase::SharedPtr m_timer;
                rclcpp::Time m_time;
                rclcpp::Publisher<ActionMsg>::SharedPtr m_action_publisher; ///< Publishes control action for actuators
                rclcpp::Subscription<TwistMsg>::SharedPtr m_twist_subscription; ///< Subscribes to twist messages
                int m_selected_test;
                std::array<double, 13> m_world_state{-3, 0, 0, 0, 0, 0, 0, 0, -3.0411, 0, 0, 0, 0};
                int m_counter;
                bool full_swing = false;
                ActionMsg m_next_action_msg;
                void publish_action();
        };

        class SysIdDisplay {
            public:
                static constexpr int width = 1024;
                static constexpr int height = 1024;
                static constexpr float framerate = 60;
                static constexpr float strafe_speed = 1.5;
                static constexpr float scale_speed = 1;

                static constexpr GLint traj_shader_cam_pos_loc = 0;
                static constexpr GLint traj_shader_cam_scale_loc = 1;
                static constexpr GLint traj_shader_color_loc = 2;

                static constexpr GLint img_shader_cam_pos_loc = 0;
                static constexpr GLint img_shader_cam_scale_loc = 1;
                static constexpr GLint img_shader_img_center_loc = 2;
                static constexpr GLint img_shader_img_width_loc = 3;
                static constexpr GLint img_shader_img_tex_loc = 4;

                void run();

            private:               
            class Trajectory {
                public:
                    Trajectory(glm::fvec4 color, float thickness, GLuint program);

                    void draw();

                    std::vector<float> vertex_buf;

                private:
                    glm::fvec4 color;
                    float thickness;
                    GLuint program;
                    GLint color_loc;
                    GLuint VBO;
                    GLuint VAO;
            };

            void init_gl(SDL_Window* window);
            void init_img();

            void draw_offset_image();

            void update_loop(SDL_Window* window);

            glm::fvec2 m_cam_pos {0.0f, 0.0f};
            float m_cam_scale = 1.0f;

            GLuint m_trajectory_shader_program;
            GLuint m_img_shader_program;

            std::unique_ptr<Trajectory> m_trajectory = nullptr;

            utils::GLObj m_offset_img_obj;
            GLuint m_offset_img_tex;

            std::vector<glm::fvec2> m_last_states;
        }
    }
}