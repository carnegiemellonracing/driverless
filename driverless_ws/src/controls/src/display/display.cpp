#include "display.hpp"

#include <chrono>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <gsl/gsl_odeiv2.h>
#include <mppi/types.cuh>
#include <utils/general_utils.hpp>


using namespace std::chrono_literals;


namespace controls {
    namespace display {

        constexpr const char *traj_vertexShaderSource = R"(
            #version 330 core
            #extension GL_ARB_explicit_uniform_location : enable

            layout (location = 0) in vec2 aPos;

            layout (location = 0) uniform vec2 camPos;
            layout (location = 1) uniform float camScale;

            void main()
            {
               vec2 sae_coords = (aPos - camPos) / camScale;
               gl_Position = vec4(sae_coords.x, sae_coords.y, 0.0f, 1.0f);
            }
        )";

        constexpr const char* traj_fragmentShaderSource = R"(
            #version 330 core
            #extension GL_ARB_explicit_uniform_location : enable

            out vec4 FragColor;

            layout (location = 2) uniform vec4 col;

            void main()
            {
               FragColor = col;
            }
        )";

        constexpr const char *img_vertex_source = R"(
            #version 330 core
            #extension GL_ARB_explicit_uniform_location : enable


            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 i_texCoord;

            layout (location = 0) uniform vec2 camPos;
            layout (location = 1) uniform float camScale;
            layout (location = 2) uniform vec2 imgCenter;
            layout (location = 3) uniform float imgWidth;

            out vec2 texCoord;

            void main()
            {
                vec2 sae_coords = (aPos * imgWidth * 0.5f + imgCenter - camPos) / camScale;
                gl_Position = vec4(sae_coords.x, sae_coords.y, 0.0f, 1.0f);
                texCoord = i_texCoord;
            }
        )";

        constexpr const char* img_fragment_source = R"(
            #version 330 core
            #extension GL_ARB_explicit_uniform_location : enable

            in vec2 texCoord;

            out vec4 FragColor;

            layout (location = 4) uniform sampler2D img;

            void main()
            {
               FragColor = texture(img, texCoord);
            }
        )";


        Display::DrawableLine::DrawableLine(glm::fvec4 color, float thickness, GLuint program)
            : color(color), program(program), thickness(thickness) {

            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);

            glBindVertexArray(VAO);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)0);
            glEnableVertexAttribArray(0);

            color_loc = glGetUniformLocation(program, "col");
        }

        void Display::DrawableLine::draw() {
            glUniform4f(color_loc, color.x, color.y, color.z, color.w);
            glLineWidth(thickness);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_buf.size(), vertex_buf.data(), GL_DYNAMIC_DRAW);

            glBindVertexArray(VAO);

            assert(vertex_buf.size() % 2 == 0);
            glDrawArrays(GL_LINE_STRIP, 0, vertex_buf.size() / 2);
        }

        
        void Display::DrawableLine::draw_points() {
            std::vector<float> triangle_points_buf = fill_triangle_points();
            glUniform4f(color_loc, color.x, color.y, color.z, color.w);
            //glLineWidth(thickness);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * triangle_points_buf.size(), triangle_points_buf.data(), GL_DYNAMIC_DRAW);

            glBindVertexArray(VAO);
            assert(triangle_points_buf.size() % 6 == 0);
            glDrawArrays(GL_TRIANGLES, 0, triangle_points_buf.size()/2);
        }

        std::vector<float> Display::DrawableLine::fill_triangle_points(){
            constexpr float radius = .25;
            constexpr float sqr3b2 = .866;
            std::vector<float> output_triangle_points;
            output_triangle_points.reserve(vertex_buf.size() * 3);
            
            for (size_t i = 0; i < vertex_buf.size()/2; i++) {
                output_triangle_points.push_back(vertex_buf[2*i]-radius*sqr3b2);
                output_triangle_points.push_back(vertex_buf[2*i+1]-radius/2);
                output_triangle_points.push_back(vertex_buf[2*i]+radius*sqr3b2);
                output_triangle_points.push_back(vertex_buf[2*i+1]-radius/2);
                output_triangle_points.push_back(vertex_buf[2*i]);
                output_triangle_points.push_back(vertex_buf[2*i+1]+radius);
                //Draw cone triangle points
                // output_triangle_points[6*i] = ;
                // output_triangle_points[6*i + 1] = ;
                // output_triangle_points[6*i + 2] = vertex_buf[2*i]+radius*sqr3b2;
                // output_triangle_points[6*i + 3] = vertex_buf[2*i+1]-radius/2;
                // output_triangle_points[6*i + 4] = vertex_buf[2*i];
                // output_triangle_points[6*i + 5] = vertex_buf[2*i+1]+radius;
                
            }
            return output_triangle_points;
        }

        Display::Display(
            std::shared_ptr<mppi::MppiController> controller,
            std::shared_ptr<state::StateEstimator> state_estimator)
                : m_controller {std::move(controller)},
                  m_state_estimator {std::move(state_estimator)} {
        }

        void Display::init_gl(SDL_Window* window) {
            m_trajectory_shader_program = utils::compile_shader(traj_vertexShaderSource, traj_fragmentShaderSource);
            m_img_shader_program = utils::compile_shader(img_vertex_source, img_fragment_source);

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glLineWidth(1.0f);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
        }

        //*******************************NEW */
        void Display::init_raceline() {
            m_raceline_line = std::make_unique<DrawableLine>(glm::fvec4 {1.0f, 0.0f, 1.0f, 1.0f}, 6, m_trajectory_shader_program);
        }


        //TODO: FIX POSITIONING
        // void Display::update_raceline() {
        //     glm::fvec2 car_pos = m_state_estimator->get_raceline_points();


        //     //not necessary. just have it so that there aren't any duplicates
        //     const float minimum_distance = 0.1f;  // so the raceline doesn't get too dense with points
        //     if (m_raceline.empty() || glm::distance(car_pos, m_race_line.back()) > minimum_distance) {
        //         m_raceline.push_back(car_pos);

        //         // if (m_raceline.size() > car_pos.size) {
        //         //     m_raceline.erase(m_raceline.begin(), m_raceline.begin() + m_raceline.size() - car_pos.size);
        //         // }
        //     }
        // }


        void Display::init_trajectories() {
            for (uint32_t i = 0; i < num_samples_to_draw; i++) {
                m_trajectories.emplace_back(glm::fvec4 {1.0f, 0.0f, 0.0f, 0.0f}, 1, m_trajectory_shader_program);
            }
        }

        void Display::init_spline() {
            m_spline = std::make_unique<DrawableLine>(glm::fvec4 {1.0f, 1.0f, 1.0f, 1.0f}, 2, m_trajectory_shader_program);
            m_left_cone_trajectory = std::make_unique<DrawableLine>(glm::fvec4 {0.0f, 0.0f, 1.0f, 1.0f}, 3, m_trajectory_shader_program);
            m_right_cone_trajectory = std::make_unique<DrawableLine>(glm::fvec4 {1.0f, 1.0f, 0.0f, 1.0f}, 3, m_trajectory_shader_program);
        }

        void Display::init_best_guess() {
            m_best_guess = std::make_unique<DrawableLine>(glm::fvec4 {0.0f, 1.0f, 0.0f, 1.0f}, 5, m_trajectory_shader_program);
        }
        

        void Display::init_img() {
            constexpr float vertices[] = {
                -1.0f, -1.0f,     0.0f, 0.0f,
                1.0f, -1.0f,      1.0f, 0.0f,
                1.0f, 1.0f,       1.0f, 1.0f,
                -1.0f, 1.0f,      0.0f, 1.0f
            };

            constexpr GLuint indices[] = {
                0, 1, 2,
                2, 3, 0
            };

            glGenVertexArrays(1, &m_offset_img_obj.vao);
            glGenBuffers(1, &m_offset_img_obj.vbo);
            glGenBuffers(1, &m_offset_img_obj.ebo);

            glBindVertexArray(m_offset_img_obj.vao);
            glBindBuffer(GL_ARRAY_BUFFER, m_offset_img_obj.vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_offset_img_obj.ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
            glEnableVertexAttribArray(1);

            glBindVertexArray(0);

            glGenTextures(1, &m_offset_img_tex);
            glBindTexture(GL_TEXTURE_2D, m_offset_img_tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glUseProgram(m_img_shader_program);
            glUniform1i(img_shader_img_tex_loc, 0);
        }

        void Display::draw_raceline() {
            assert(m_raceline_line!=nullptr);
            m_raceline_line->vertex_buf = std::vector<float>(m_raceline_points.size() * 2);

            for (size_t i = 0; i < m_raceline_points.size(); i++) {
                m_raceline_line->vertex_buf[2 * i] = m_raceline_points[i].x;
                m_raceline_line->vertex_buf[2 * i + 1] = m_raceline_points[i].y;

                // as raceline ages changes color
                // float alpha = static_cast<float>(i) / m_raceline_points.size();
                // m_raceline_line->color.w = alpha;
            }

            m_raceline_line->draw();

        }


        void Display::fill_trajectories() {
            using namespace glm;

            const auto& states = m_last_state_trajectories;

            for (uint32_t i = 0; i < num_samples_to_draw; i++) {
                if (m_trajectories[i].vertex_buf.size() < num_timesteps) {
                    m_trajectories[i].vertex_buf = std::vector<float> (num_timesteps * 2);
                }

                for (uint32_t j = 0; j < num_timesteps; j++) {
                    uint32_t state_idx = i * state_dims * num_timesteps + j * state_dims;
                    m_trajectories[i].vertex_buf[2 * j] = states[state_idx];
                    m_trajectories[i].vertex_buf[2 * j + 1] = states[state_idx + 1];
                }
            }
        }

        void Display::draw_trajectories() {
            glUseProgram(m_trajectory_shader_program);
            for (uint32_t i = 0; i < num_samples_to_draw; i++) {
                DrawableLine& t = m_trajectories[i];
                t.draw();
            }
        }

        void Display::draw_spline() {
            const auto& frames = m_spline_frames;

            assert(m_spline != nullptr);
            m_spline->vertex_buf = std::vector<float>(frames.size() * 2);
            for (size_t i = 0; i < frames.size(); i++) {
                m_spline->vertex_buf[2 * i] = frames[i].x;
                m_spline->vertex_buf[2 * i + 1] = frames[i].y;
            }

            m_spline->draw();
        }

        void Display::draw_cones()
        {
            assert(m_left_cone_trajectory != nullptr);
            assert(m_right_cone_trajectory != nullptr);
            const auto& left_cone_points = m_all_left_cone_points;
            const auto& right_cone_points = m_all_right_cone_points;
            m_left_cone_trajectory->vertex_buf = std::vector<float>(left_cone_points.size() * 2);
            for (size_t i = 0; i < left_cone_points.size(); i++) {
                //Draw trajectory line
                m_left_cone_trajectory->vertex_buf[2 * i] = m_all_left_cone_points[i].x;
                m_left_cone_trajectory->vertex_buf[2 * i + 1] = m_all_left_cone_points[i].y;
            }

            m_right_cone_trajectory->vertex_buf = std::vector<float>(right_cone_points.size() * 2);
            for (size_t i = 0; i < right_cone_points.size(); i++) {
                // Draw trajectory line
                m_right_cone_trajectory->vertex_buf[2 * i] = right_cone_points[i].x;
                m_right_cone_trajectory->vertex_buf[2 * i + 1] = right_cone_points[i].y;
            }


            m_left_cone_trajectory->draw();
            m_right_cone_trajectory->draw();
            m_left_cone_trajectory->draw_points();
            m_right_cone_trajectory->draw_points();
        }

        void Display::draw_best_guess() {
            const auto& frames = m_last_reduced_state_trajectory;

            assert(m_best_guess != nullptr);
            m_best_guess->vertex_buf = std::vector<float>(frames.size() * 2);
            for (size_t i = 0; i < frames.size(); i++) {
                m_best_guess->vertex_buf[2 * i] = frames[i].x;
                m_best_guess->vertex_buf[2 * i + 1] = frames[i].y;
            }

            m_best_guess->draw();
        }

        void Display::draw_offset_image() {
            // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

            auto& offset_image = m_offset_image;
            // auto max = std::max_element(offset_image.pixels.begin(), offset_image.pixels.end());

            // for (int i = 0; i < offset_image.pixels.size(); i++) {
            //     offset_image.pixels[i] /= *max;
            // }

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, m_offset_img_tex);
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA,
                offset_image.pix_width, offset_image.pix_height,
                0, GL_RGBA, GL_FLOAT, offset_image.pixels.data()
            );

            glUseProgram(m_img_shader_program);
            glUniform2f(img_shader_cam_pos_loc, m_cam_pos.x, m_cam_pos.y);
            glUniform1f(img_shader_cam_scale_loc, m_cam_scale);
            glUniform2f(img_shader_img_center_loc, offset_image.center.x, offset_image.center.y);
            glUniform1f(img_shader_img_width_loc, offset_image.world_width);

            glBindVertexArray(m_offset_img_obj.vao);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
        void Display::draw_triangles() {
            std::vector<float> triangle_vertices = m_state_estimator->get_vertices();
            unsigned int VAO;
            unsigned int VBO;
            glGenBuffers(1,&VBO);
            glGenVertexArrays(1,&VAO);
            // glUniform4f(color_loc, color.x, color.y, color.z, color.w);
            // glLineWidth(thickness);
            glUseProgram(m_img_shader_program);
            glBindVertexArray(VAO);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * triangle_vertices.size(), triangle_vertices.data(), GL_DYNAMIC_DRAW);

            assert(triangle_vertices.size() % 3 == 0);
            glDrawArrays(GL_TRIANGLES, 0, triangle_vertices.size() / 3);
        }

        void Display::draw_car(){
            //Make sure a line exists
            assert(m_best_guess != nullptr);
            assert(m_best_guess->vertex_buf.size() != 0);
            //In (x,y) coords, (0,1) is front left, (2,3) is front right, (4,5) is back left, (6,7) is back right
            std::vector<float> carpts = std::vector<float>(8);
            carpts.push_back(cg_to_nose);
            carpts.push_back(cg_to_side);
            carpts.push_back(cg_to_nose);
            carpts.push_back(-cg_to_side);
            carpts.push_back(-cg_to_rear);
            carpts.push_back(cg_to_side);
            carpts.push_back(-cg_to_rear);
            carpts.push_back(-cg_to_side);
            unsigned int VAO;
            unsigned int VBO;
            glGenBuffers(1,&VBO);
            glGenVertexArrays(1,&VAO);
            GLint color_loc = glGetUniformLocation(m_img_shader_program, "col");
            glUniform4f(color_loc, 1, 0, 0, 1);
            glUseProgram(m_img_shader_program);
            glBindVertexArray(VAO);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * carpts.size(), carpts.data(), GL_DYNAMIC_DRAW);

            //assert(triangle_vertices.size() % 3 == 0);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 2);

        }

        void Display::run() {
            SDL_Window* window = utils::create_sdl2_gl_window("MPPI Display", width, height);
            init_gl(window);
            init_trajectories();
            init_spline();
            init_best_guess();
            init_img();

            init_raceline();

            update_loop(window);
        }

        void Display::update_loop(SDL_Window* window) {
            bool should_close = false;
            auto last_time = std::chrono::system_clock::now();
            float delta_time = 1 / framerate;
            while (!should_close) {
                SDL_Event e;
                while (SDL_PollEvent(&e) > 0) {
                    switch (e.type) {
                        case SDL_QUIT:
                            should_close = true;
                            break;

                        default:
                            break;
                    }
                }

                const uint8_t* keyboard_state = SDL_GetKeyboardState(nullptr);

                if (keyboard_state[SDL_SCANCODE_LEFT]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(-1,0);
                }
                if (keyboard_state[SDL_SCANCODE_RIGHT]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(1,0);
                }
                if (keyboard_state[SDL_SCANCODE_UP]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(0,1);
                }
                if (keyboard_state[SDL_SCANCODE_DOWN]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(0,-1);
                }

                if (keyboard_state[SDL_SCANCODE_S]) {
                    m_cam_scale *= pow(1 + scale_speed, delta_time);
                }
                if (keyboard_state[SDL_SCANCODE_W]) {
                    m_cam_scale *= pow(1 / (1 + scale_speed), delta_time);
                }

                glUseProgram(m_trajectory_shader_program);
                glUniform2f(traj_shader_cam_pos_loc, m_cam_pos.x, m_cam_pos.y);
                glUniform1f(traj_shader_cam_scale_loc, m_cam_scale);

                glClear(GL_COLOR_BUFFER_BIT);

                m_spline_frames = m_state_estimator->get_spline_frames();
                m_all_left_cone_points = m_state_estimator->get_all_left_cone_points();
                m_all_right_cone_points = m_state_estimator->get_all_right_cone_points();

                m_left_cone_points = m_state_estimator->get_left_cone_points();
                m_right_cone_points = m_state_estimator->get_right_cone_points();
                m_raceline_points = m_state_estimator->get_raceline_points();

                m_state_estimator->get_offset_pixels(m_offset_image);
                m_last_reduced_state_trajectory = m_controller->last_reduced_state_trajectory();
                m_last_state_trajectories = m_controller->last_state_trajectories(num_samples_to_draw);
                
                m_left_cone_trajectory->vertex_buf = std::vector<float>(m_left_cone_points.size() * 2);
                for (size_t i = 0; i < m_left_cone_points.size(); i++) {
                    m_left_cone_trajectory->vertex_buf[2 * i] = m_left_cone_points[i].x;
                    m_left_cone_trajectory->vertex_buf[2 * i + 1] = m_left_cone_points[i].y;
                }

                m_right_cone_trajectory->vertex_buf = std::vector<float>(m_right_cone_points.size() * 2);
                for (size_t i = 0; i < m_right_cone_points.size(); i++) {
                    m_right_cone_trajectory->vertex_buf[2 * i] = m_right_cone_points[i].x;
                    m_right_cone_trajectory->vertex_buf[2 * i + 1] = m_right_cone_points[i].y;
                }


                draw_offset_image();

                fill_trajectories();
                draw_trajectories();

                draw_spline();
                draw_cones();
                draw_best_guess();
                //draw_car();

                draw_raceline();
                //update_raceline();

                SDL_GL_SwapWindow(window);

                auto now = std::chrono::system_clock::now();
                uint32_t to_go_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
                SDL_Delay(std::max(0u, to_go_ms));

                now = std::chrono::system_clock::now();
                delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count() / 1000.0f;
                last_time = now;
            }

            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }

    }
}
