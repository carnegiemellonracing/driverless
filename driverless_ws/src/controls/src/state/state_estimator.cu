/**
 * To-Do list:
 * - Add drawing of small triangles corresponding to cones
 * - Automatic zoom-out/pan
 * - car boundary
 * 
 */


#ifndef GLM_FORCE_QUAT_DATA_WXYZ
#define GLM_FORCE_QUAT_DATA_WXYZ
#endif

#include <utils/cuda_utils.cuh>
#include <utils/gl_utils.hpp>
#include <cuda_globals/cuda_globals.cuh>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cuda_constants.cuh>
#include <cmath>
#include <cuda_gl_interop.h>


#include "state_estimator.cuh"
#include "state_estimator.hpp"

#include <iosfwd>
#include <vector>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include <glm/common.hpp>
#include <mppi/functors.cuh>
#include <SDL2/SDL_video.h>

#include <midline/svm_conv.hpp>
#include <utils/ros_utils.hpp>
#include <utils/cuda_macros.cuh>

namespace {
    // Helper function for timing with optional GPU synchronization
    template<bool EnableSync>
    std::chrono::time_point<std::chrono::high_resolution_clock> sync_now() {
        // if constexpr (EnableSync) {
        //     glFinish(); // Ensure GPU has completed all work
        // }
        return std::chrono::high_resolution_clock::now();
    }
    
    // Helper function to log timings if enabled
    template<bool EnableLog>
    void log_timings(
        rclcpp::Logger& logger,
        float total_ms,
        float gl_context_ms,
        float tex_info_ms,
        float buffer_cones_ms,
        float buffer_spline_ms,
        float unmap_ms,
        float fake_track_ms,
        float curv_frame_ms,
        float map_ms,
        float sync_state_ms,
        float sync_tex_ms,
        float display_ms,
        float unbind_ms) {
        
        if constexpr (EnableLog) {
            // Note: When EnableLog==true with glFinish calls, these timings include GPU execution time
            RCLCPP_INFO(logger, 
                "render_and_sync timing (ms): Total=%.2f, GL context=%.2f, Tex info=%.2f, "
                "Buffer cones=%.2f, Buffer spline=%.2f, Unmap=%.2f, Fake track=%.2f, "
                "Curv frame=%.2f, Map=%.2f, Sync state=%.2f, Sync tex=%.2f, Display=%.2f, Unbind=%.2f",
                total_ms, gl_context_ms, tex_info_ms,
                buffer_cones_ms, buffer_spline_ms, unmap_ms,
                fake_track_ms, curv_frame_ms, map_ms,
                sync_state_ms, sync_tex_ms, display_ms, unbind_ms);
            
            // Log sum of component times to verify accounting
            float sum = gl_context_ms + tex_info_ms + buffer_cones_ms + buffer_spline_ms + 
                      unmap_ms + fake_track_ms + curv_frame_ms + map_ms + 
                      sync_state_ms + sync_tex_ms + unbind_ms;
            
            RCLCPP_INFO(logger, "Sum of component times: %.2f ms (%.2f%% of total)",
                       sum, (sum / total_ms) * 100.0f);
        }
    }
}

namespace controls {
    namespace state {
 

        // State Estimator

        std::shared_ptr<StateEstimator> StateEstimator::create(std::mutex& mutex, LoggerFunc logger) {
            return std::make_shared<StateEstimator_Impl>(mutex, logger);
        }

        // StateEstimator_Impl helpers


        /**
         * @brief GPU shader code. In a string because historically, shaders are JIT compiled.
         * Runs in parallel for every vertex in the VBO, similar to a functor.
         * Layouts initially specified in gen_gl_path().
         *
         * Transforms from vertex IRL position to clip space (rendering coordinate frame) using scale
         * and center. For now this transformation is unneeded. However, suppose with SLAM, (x,y) from
         * path planning are from a stationary world perspective, but we want to render to the car's vicinity.
         *
         * @note: the z coordinate in gl_Position is how we exploit depth testing to break ties between multiple
         * overlapping triangles.
         * note that i_world_pos is a vec2 so the dimensions in gl_Position check out (shader languages are built for vector math)
         * uniform is similar to __constant__ in CUDA
         * @note: far_frustrum is for better precision since only relative ordering matters
         * @note: we don't use the 4th coordinate of gl_Position, but it is needed. Look up homogenous coordinates.
         *
         * @param[in] i_world_pos x, y from path planning
         * @param[in] i_curv_pose progress, offset, heading
         * @param[out] o_curv_pose same as i_curv_pose, passed along
         * @return gl_Position
         */
        constexpr const char* vertex_source_fake_track = R"(
            #version 330 core
            #extension GL_ARB_explicit_uniform_location : enable

            layout (location = 0) in vec2 i_world_pos;
            layout (location = 1) in vec3 i_curv_pose;

            out vec3 o_curv_pose;

            layout (location = 0) uniform float scale;
            layout (location = 1) uniform vec2 center;

            const float far_frustum = 10.0f;

            void main() {
                gl_Position = vec4(scale * (i_world_pos - center), abs(i_curv_pose.y) / far_frustum, 1.0);
                o_curv_pose = i_curv_pose;
            }
        )";

        /**
         * @brief GPU shader code. "fragment shader"
         * Runs in parallel for every pixel in the triangles.
         * We only manually calculated o_curv_pose for the vertices,
         * interpolation for the pixels in between happens here automatically.
         * 4th color (1.0f) represents in bounds, compared to background which has -1.0 representing OOB
         * @return FragColor The color of each foreground pixel
         */
        constexpr const char* fragment_source_fake_track = R"(
            #version 330 core

            in vec3 o_curv_pose;

            out vec4 FragColor;

            void main() {
                FragColor = vec4(o_curv_pose, 1.0f);
            }
        )";

        // TODO: I think abs(i_curv_pose.y) is not needed, and hence i_curv_pose is not needed, because glLineStrip
        // means there will be no overlapping triangles
        // besides cones don't have offset information
        constexpr const char *vertex_source = R"(
            #version 330 core
            #extension GL_ARB_explicit_uniform_location : enable

            layout (location = 0) in vec2 i_world_pos;
            layout (location = 1) in vec3 i_curv_pose;

            out vec2 o_world_pose;

            layout (location = 0) uniform float scale;
            layout (location = 1) uniform vec2 center;

            const float far_frustum = 10.0f;

            void main() {
                o_world_pose = scale * (i_world_pos - center);
                gl_Position = vec4(scale * (i_world_pos - center), abs(i_curv_pose.y) / far_frustum, 1.0);
            }
        )";

        constexpr const char *fragment_source = R"(
            #version 330 core

            in vec2 o_world_pose;

            out vec4 FragColor;

            uniform sampler2D fake_track_texture;

            void main() {
                FragColor = texture(fake_track_texture, o_world_pose / 2.0 + 0.5); // convert from normalized device coordinates to texture coordinates
            }
        )";

        constexpr const char *fragment_source_double_cone = R"(
            #version 330 core

            in vec2 o_world_pose;

            out vec4 FragColor;

            uniform sampler2D left_track_texture;
            uniform sampler2D right_track_texture;

            void main() {
                vec4 color1 = texture(left_track_texture, o_world_pose / 2.0 + 0.5);
                vec4 color2 = texture(right_track_texture, o_world_pose / 2.0 + 0.5);
                FragColor = vec4(color1.x, 0, color2.x, color1.w);
            }
        )";

        // methods

        // Constructor
        StateEstimator_Impl::StateEstimator_Impl(std::mutex& mutex, LoggerFunc logger)
            : m_mutex {mutex}, m_logger {logger}, m_logger_obj {rclcpp::get_logger("")} {
            std::lock_guard<std::mutex> guard {mutex};

            m_logger("initializing state estimator");

#ifdef DISPLAY
            if (display_on) {
                m_gl_window = utils::create_sdl2_gl_window(
                    "Spline Frame Lookup", curv_frame_lookup_tex_width, curv_frame_lookup_tex_width,
                    0, &m_gl_context
                );
            } else {
                // dummy window to create opengl context for curv frame buffer
                m_gl_window = utils::create_sdl2_gl_window(
                    "Spline Frame Lookup Dummy", 1, 1,
                    SDL_WINDOW_HIDDEN, &m_gl_context
                );
            }
#else
            m_gl_window = utils::create_sdl2_gl_window(
                "Spline Frame Lookup Dummy", 1, 1,
                SDL_WINDOW_HIDDEN, &m_gl_context
            );
#endif

            m_logger("making state estimator gl context current");
            utils::make_gl_current_or_except(m_gl_window, m_gl_context);

            m_logger("compiling state estimator shaders");
            
            m_fake_track_shader_program = utils::compile_shader(vertex_source_fake_track, fragment_source_fake_track);
            m_curv_frame_lookup_shader_program = utils::compile_shader(vertex_source, fragment_source);
            m_double_cone_shader_program = utils::compile_shader(vertex_source, fragment_source_double_cone);
            m_logger("setting state estimator gl properties");

            glViewport(0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width);

            m_logger("generating state estimator gl buffers");
            gen_curv_frame_lookup_framebuffer();
            gen_gl_path(m_gl_path);
            gen_fake_track(m_midline_fake_track);
            if (no_midline_controller) {
                gen_fake_track(m_left_fake_track);
                gen_fake_track(m_right_fake_track);
            }




            glFinish();
            utils::make_gl_current_or_except(m_gl_window, nullptr);
            m_logger("finished state estimator initialization");
            SDL_GLContext curr_context = SDL_GL_GetCurrentContext();
            SDL_Window* curr_window = SDL_GL_GetCurrentWindow();
            // RCLCPP_INFO(m_logger_obj, "After constructor: window: %p, context %p", curr_window, curr_context);


        }

        void StateEstimator_Impl::gen_curv_frame_lookup_framebuffer() {
            glGenFramebuffers(1, &m_curv_frame_lookup_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, m_curv_frame_lookup_fbo);

            glGenRenderbuffers(1, &m_curv_frame_lookup_rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, m_curv_frame_lookup_rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_curv_frame_lookup_rbo);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                throw std::runtime_error("Framebuffer is not complete");
            }
            // reset framebuffer to default
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        void StateEstimator_Impl::gen_fake_track(FakeTrackInfo& fake_track_info) {
            // generate the framebuffer for the fake track
            glGenFramebuffers(1, &fake_track_info.fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fake_track_info.fbo);

            // generate texture
            glGenTextures(1, &fake_track_info.texture_color);
            glBindTexture(GL_TEXTURE_2D, fake_track_info.texture_color);

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width, 0, GL_RGBA, GL_FLOAT, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);

            // attach it to currently bound framebuffer object
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fake_track_info.texture_color, 0); 
            
            GLuint depth_rbo;

            glGenRenderbuffers(1, &depth_rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, curv_frame_lookup_tex_width,  curv_frame_lookup_tex_width);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                throw std::runtime_error("Fake track framebuffer is not complete");
            }
            // reset framebuffer to default
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            gen_gl_path(fake_track_info.path);
        }

        StateEstimator_Impl::~StateEstimator_Impl() {
            // utils::sync_gl_and_unbind_context(m_gl_window);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }

        std::vector<glm::fvec2> process_ros_points(std::vector<geometry_msgs::msg::Point,
                                                               std::allocator<geometry_msgs::msg::Point>>
                                                       points)
        {
            std::vector<glm::fvec2> processed_points;
            processed_points.reserve(points.size());
            for (const auto &point : points)
            {
                float cone_y = static_cast<float>(point.y);
                float cone_x = static_cast<float>(point.x);
                processed_points.push_back(
                    glm::fvec2(cone_x, cone_y));
            }
            paranoid_assert(processed_points.size() == points.size());
            return processed_points;
        }

        void StateEstimator_Impl::on_spline(const SplineMsg& spline_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};

            m_logger("beginning state estimator spline processing");

            paranoid_assert(spline_msg.frames.size() > 0);

            if (ingest_midline) {
                m_spline_frames = process_ros_points(spline_msg.frames);
            }

            m_logger("finished state estimator spline processing");
        }

        void StateEstimator_Impl::on_quat(const QuatMsg& quat_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};
            const float yaw = quat_msg_to_yaw(quat_msg);
            m_state_projector.record_yaw(yaw, quat_msg.header.stamp);
        }

        void StateEstimator_Impl::on_position_lla(const PositionLLAMsg& position_lla_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};
            const auto pair = position_lla_msg_to_xy(position_lla_msg);
            m_state_projector.record_position_lla(pair.first, pair.second, position_lla_msg.header.stamp);
        }

        static SplineMsg spline_frames_to_msg(std::vector<glm::fvec2> frames_vec) {
            SplineMsg frames_msg;
            frames_msg.frames.reserve(frames_vec.size());
            for (auto& frame : frames_vec) {
                geometry_msgs::msg::Point point;
                point.x = frame.x;
                point.y = frame.y;
                frames_msg.frames.push_back(point);
            }
            return frames_msg;
        }

        static float compute_score(size_t num_blue_cones, size_t num_yellow_cones, rclcpp::Duration time_diff) {
            const size_t smaller_num_cones = std::min(num_blue_cones, num_yellow_cones);
            const size_t total_num_cones = num_blue_cones + num_yellow_cones;
            const float num_seconds = time_diff.nanoseconds() / 1e9f;
            const float score = smaller_num_cones_multiplier * smaller_num_cones + total_num_cones_multiplier * total_num_cones + seconds_since_spline_multiplier * num_seconds;
            return score;
        }


        std::optional<std::pair<std::vector<glm::fvec2>, float>> StateEstimator_Impl::get_best_old_spline(rclcpp::Time current_time, State final_delta_state) {
            if (m_spline_history.size() == 0) {
                return std::nullopt;
            }
            std::vector<glm::fvec2> best_spline;
            float best_score = FLT_MIN;
            std::deque<StateEstimator_Impl::OldSpline>::iterator best_spline_iter;
            for (auto iter = m_spline_history.begin(); iter != m_spline_history.end(); iter++) {
                StateEstimator_Impl::OldSpline old_spline = *iter;
                const auto time_diff = current_time - old_spline.timestamp;
                const float score = compute_score(old_spline.num_blue_cones, old_spline.num_yellow_cones, time_diff);
                if (score > best_score) {
                    best_spline = old_spline.old_spline_frames;
                    best_score = score;
                    best_spline_iter = iter;
                }

            }

            // Now we need to compute the transformation from best_spline to the current frame
            best_spline_iter++;
            State aggregate_delta_state;
            float last_absolute_yaw = M_PI_2;
            for (auto iter = best_spline_iter; iter != m_spline_history.end(); iter++) {
                StateEstimator_Impl::OldSpline old_spline = *iter;
                State relative_state_to_previous = old_spline.relative_state_to_previous;
                float yaw_difference = relative_state_to_previous[state_yaw_idx] - M_PI_2;
                glm::fvec2 state_difference = rotate_point({relative_state_to_previous[state_x_idx], 
                    relative_state_to_previous[state_y_idx]}, last_absolute_yaw - M_PI_2);

                aggregate_delta_state[state_x_idx] += state_difference.x;
                aggregate_delta_state[state_y_idx] += state_difference.y;
                aggregate_delta_state[state_yaw_idx] += yaw_difference;

                last_absolute_yaw += yaw_difference;
            }

            glm::fvec2 state_difference = rotate_point({final_delta_state[state_x_idx], 
                final_delta_state[state_y_idx]}, last_absolute_yaw - M_PI_2);
            aggregate_delta_state[state_x_idx] += state_difference.x;
            aggregate_delta_state[state_y_idx] += state_difference.y;
            aggregate_delta_state[state_yaw_idx] += final_delta_state[state_yaw_idx];

            // Now given aggregate delta state, I need to transform the best_spline into the current frame
            std::vector<glm::fvec2> best_spline_frames_transformed;
            best_spline_frames_transformed.reserve(best_spline.size());
            
            for (auto& p : best_spline) {
                glm::fvec2 rel_point = p - glm::fvec2{aggregate_delta_state[state_x_idx], aggregate_delta_state[state_y_idx]};
                glm::fvec2 rotated_point = rotate_point(rel_point, -aggregate_delta_state[state_yaw_idx]);
                best_spline_frames_transformed.push_back(rotated_point);
            }
            
            return std::make_pair(best_spline_frames_transformed, best_score);
        } 


        float StateEstimator_Impl::on_cone(const ConeMsg& cone_msg, rclcpp::Publisher<SplineMsg>::SharedPtr cone_publisher) {
            std::lock_guard<std::mutex> guard {m_mutex};

            paranoid_assert(cone_msg.blue_cones.size() > 0);
            paranoid_assert(cone_msg.yellow_cones.size() > 0);

            m_logger("beginning state estimator cone processing");

            m_left_cone_points.clear();
            m_right_cone_points.clear();

            m_left_cone_points = process_ros_points(cone_msg.blue_cones);
            m_right_cone_points = process_ros_points(cone_msg.yellow_cones);

            float svm_time = 0.0f;



            // I need this above the spline calculation because I want to get the delta state
            std::optional<State> delta_state_opt;
            if constexpr (reset_pose_on_cone) {
                delta_state_opt = m_state_projector.record_pose(0, 0, M_PI_2, cone_msg.header.stamp);
            }

            // Might also need to && delta_state_opt.has_value()
            if (!ingest_midline && !no_midline_controller) {
                assert(delta_state_opt.has_value()); 

                // Come up with a way to weight between 
                const float this_cone_score = compute_score(cone_msg.blue_cones.size(), cone_msg.yellow_cones.size(), rclcpp::Duration::from_seconds(0));
                const std::optional<std::pair<std::vector<glm::vec2>, float>> best_spline_and_score = get_best_old_spline(cone_msg.header.stamp);
                if (best_spline_and_score.has_value() && best_spline_and_score.value().second > this_cone_score) {
                    m_spline_frames = best_spline_and_score.value().first;
                    svm_time = 0.0f;
                } else {

                    midline::Cones cones;
                    for (const auto& cone : m_left_cone_points) {
                        cones.addBlueCone(cone.x, cone.y, 0);
                    }
                    for (const auto& cone : m_right_cone_points) {
                        cones.addYellowCone(cone.x, cone.y, 0);
                    }

                    // // TODO: convert this to using std::transform
                    auto svm_start = std::chrono::high_resolution_clock::now();            
                    auto spline_frames = midline::svm_test::cones_to_midline(cones);
                    // auto spline_frames = midline::svm_slow::cones_to_midline(cones);
                    // auto _ = midline::svm_test::cones_to_midline(cones);
                    auto svm_end = std::chrono::high_resolution_clock::now();
                    svm_time = std::chrono::duration_cast<std::chrono::milliseconds>(svm_end - svm_start).count();
                    m_spline_frames.clear();
                    for (const auto& frame : spline_frames) {
                        paranoid_assert(!isnan(frame.first) && !isnan(frame.second));
                        m_spline_frames.emplace_back(frame.first, frame.second);
                    }
                    if (m_spline_frames.size() < 2) {
                        paranoid_assert(cone_msg.blue_cones.size() == 0 && cone_msg.yellow_cones.size() == 0);
                    }
                }
                if (publish_spline) {
                    cone_publisher->publish(spline_frames_to_msg(m_spline_frames));
                }

                OldSpline old_spline;
                old_spline.old_spline_frames = m_spline_frames;
                old_spline.relative_state_to_previous = delta_state_opt.value();
                old_spline.num_blue_cones = cone_msg.blue_cones.size();
                old_spline.num_yellow_cones = cone_msg.yellow_cones.size();
                old_spline.timestamp = cone_msg.header.stamp;
                m_spline_history.push_back(old_spline);
                if (m_spline_history.size() > maximum_spline_history_length) {
                    m_spline_history.pop_front();
                }
                
            }

#ifdef DISPLAY
            if (display_on) {
                m_all_left_cone_points.clear();
                m_all_right_cone_points.clear();

                m_all_left_cone_points = process_ros_points(cone_msg.orange_cones);
                m_all_right_cone_points = process_ros_points(cone_msg.unknown_color_cones);
                m_raceline_points = process_ros_points(cone_msg.big_orange_cones);

            }
#endif

            
            m_logger("finished state estimator cone processing");
            return svm_time;
        }

        void StateEstimator_Impl::on_twist(const TwistMsg &twist_msg, const rclcpp::Time &time) {
            // TODO: whats up with all these mutexes
            std::lock_guard<std::mutex> guard {m_mutex};

            const float speed = twist_msg_to_speed(twist_msg);

            m_state_projector.record_speed(speed, time);

        }

        void StateEstimator_Impl::on_pose(const PoseMsg &pose_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};
            m_state_projector.record_pose(
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.orientation.z,
                pose_msg.header.stamp);
        }


        void StateEstimator_Impl::record_control_action(const Action& action, const rclcpp::Time& time) {
            std::lock_guard<std::mutex> guard {m_mutex};
            // TODO: put this under a constexpr switch flag, now we always record for debugging purposes
            m_state_projector.record_action(action, rclcpp::Time{
                                                        time.nanoseconds() + static_cast<int64_t>(approx_propogation_delay * 1e9f),
                                                        default_clock_type});

        }

        void StateEstimator_Impl::render_and_sync(State state) {
            if (no_midline_controller && m_follow_midline_only) {
                throw std::runtime_error("No mildine controller + follow midline only doesn't make any sense");
            }
            std::lock_guard<std::mutex> guard {m_mutex};
            
            // Timing variables
            auto start_time = sync_now<log_render_and_sync_timing>();
            auto last_time = start_time;
            std::chrono::duration<float, std::milli> gl_context_time, tex_info_time, 
                buffer_cones_time, buffer_spline_time, unmap_time, 
                fake_track_time, curv_frame_time, map_time, 
                sync_state_time, sync_tex_time, display_time, unbind_time, total_time;
            

            SDL_GLContext log_context;
            SDL_Window* log_window;
            if constexpr (log_render_and_sync_timing) {
                log_context = SDL_GL_GetCurrentContext();
                log_window = SDL_GL_GetCurrentWindow();
                RCLCPP_INFO(m_logger_obj, "Before start of render and sync: window: %p, context %p", log_window, log_context);
            }



            // enable openGL
            utils::make_gl_current_or_except(m_gl_window, m_gl_context);
            if constexpr (log_render_and_sync_timing) {
                RCLCPP_INFO(m_logger_obj, "Call to make gl current or except: window: %p, context %p", m_gl_window, m_gl_context);
            }
            

            // // exclusively for logging
            if constexpr (log_render_and_sync_timing) {
                log_context = SDL_GL_GetCurrentContext();
                log_window = SDL_GL_GetCurrentWindow();
                RCLCPP_INFO(m_logger_obj, "After make gl current in render and sync - window: %p, context %p", log_window, log_context);
            }


            auto current_time = sync_now<log_render_and_sync_timing>();
            gl_context_time = current_time - last_time;
            last_time = current_time;

            m_logger("generating spline frame lookup texture info...");
            gen_tex_info({state[state_x_idx], state[state_y_idx]});
            current_time = sync_now<log_render_and_sync_timing>();
            tex_info_time = current_time - last_time;
            last_time = current_time;

            m_logger("filling OpenGL buffers...");
            // takes car position, places them in the vertices
            fill_path_buffers_cones();
            current_time = sync_now<log_render_and_sync_timing>();
            buffer_cones_time = current_time - last_time;
            last_time = current_time;
            
            if (!no_midline_controller) {
                fill_path_buffers_for_fake_track(m_midline_fake_track, m_spline_frames);
            } else {
                fill_path_buffers_for_fake_track(m_left_fake_track, m_left_cone_points);
                fill_path_buffers_for_fake_track(m_right_fake_track, m_right_cone_points);
            }
            current_time = sync_now<log_render_and_sync_timing>();
            buffer_spline_time = current_time - last_time;
            last_time = current_time;

            m_logger("unmapping CUDA curv frame lookup texture for OpenGL rendering");
            unmap_curv_frame_lookup();
            current_time = sync_now<log_render_and_sync_timing>();
            unmap_time = current_time - last_time;
            last_time = current_time;

            // render the lookup table
            m_logger("rendering curv frame lookup table...");
            if (!no_midline_controller) {
                render_fake_track(m_midline_fake_track);
            } else {
                render_fake_track(m_left_fake_track);
                render_fake_track(m_right_fake_track);
            }

            current_time = sync_now<log_render_and_sync_timing>();
            fake_track_time = current_time - last_time;
            last_time = current_time;
            
            if (!m_follow_midline_only) {
                render_curv_frame_lookup();
                current_time = sync_now<log_render_and_sync_timing>();
                curv_frame_time = current_time - last_time;
                last_time = current_time;
            }

            m_logger("mapping OpenGL curv frame texture back to CUDA");
            map_curv_frame_lookup();
            current_time = sync_now<log_render_and_sync_timing>();
            map_time = current_time - last_time;
            last_time = current_time;

            m_logger("syncing world state to device");
            CUDA_CALL(cudaMemcpyToSymbol(cuda_globals::curr_state, state.data(), state_dims * sizeof(float)));
            current_time = sync_now<log_render_and_sync_timing>();
            sync_state_time = current_time - last_time;
            last_time = current_time;

            m_logger("syncing spline frame lookup texture info to device");
            sync_tex_info();
            current_time = sync_now<log_render_and_sync_timing>();
            sync_tex_time = current_time - last_time;
            last_time = current_time;


#ifdef DISPLAY
            if (display_on) {
                m_last_offset_image.pixels = std::vector<float>(4 * curv_frame_lookup_tex_width * curv_frame_lookup_tex_width);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, m_curv_frame_lookup_fbo);
                glReadPixels(
                    0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width,
                    GL_RGBA, GL_FLOAT,
                    m_last_offset_image.pixels.data()
                );

                m_last_offset_image.pix_width = curv_frame_lookup_tex_width;
                m_last_offset_image.pix_height = curv_frame_lookup_tex_width;
                m_last_offset_image.center = {m_curv_frame_lookup_tex_info.xcenter, m_curv_frame_lookup_tex_info.ycenter};
                m_last_offset_image.world_width = m_curv_frame_lookup_tex_info.width;

            }
#endif
            current_time = sync_now<log_render_and_sync_timing>();
            display_time = current_time - last_time;
            last_time = current_time;

            utils::sync_gl_and_unbind_context(m_gl_window);
            if constexpr (log_render_and_sync_timing) {
                log_context = SDL_GL_GetCurrentContext();
                log_window = SDL_GL_GetCurrentWindow();
                RCLCPP_INFO(m_logger_obj, "What happens after unbinding?: window: %p, context %p", log_window, log_context);
            }

            current_time = sync_now<log_render_and_sync_timing>();
            unbind_time = current_time - last_time;
            
            // Total time
            total_time = current_time - start_time;
            
            // Log timing information using our helper function
            log_timings<log_render_and_sync_timing>(
                m_logger_obj,
                total_time.count(),
                gl_context_time.count(),
                tex_info_time.count(),
                buffer_cones_time.count(),
                buffer_spline_time.count(),
                unmap_time.count(),
                fake_track_time.count(),
                curv_frame_time.count(),
                map_time.count(),
                sync_state_time.count(),
                sync_tex_time.count(),
                display_time.count(),
                unbind_time.count()
            );
        }

        std::optional<State> StateEstimator_Impl::project_state(const rclcpp::Time& time) {
            std::lock_guard<std::mutex> guard {m_mutex};
            auto state = m_state_projector.project(
                rclcpp::Time{
                    time.nanoseconds() + static_cast<int64_t>((approx_propogation_delay + approx_mppi_time) * 1e9f),
                    default_clock_type},
                m_logger);
                 
            return state;
        }


        bool StateEstimator_Impl::is_ready() {
            std::lock_guard<std::mutex> guard {m_mutex};

            return m_state_projector.is_ready();
        }

        void StateEstimator_Impl::set_logger(LoggerFunc logger) {
            std::lock_guard<std::mutex> guard {m_mutex};

            m_logger = logger;
        }

        void StateEstimator_Impl::set_logger_obj(rclcpp::Logger logger)
        {
            std::lock_guard<std::mutex> guard {m_mutex};
            m_logger_obj = logger;
        }

        std::vector<glm::fvec2> StateEstimator_Impl::get_spline_frames()
        {
            std::lock_guard<std::mutex> guard{m_mutex};

            std::vector<glm::fvec2> res(m_spline_frames.size());
            for (size_t i = 0; i < m_spline_frames.size(); i++)
            {
                res[i] = {m_spline_frames[i].x, m_spline_frames[i].y};
            }
            return res;
        }

#ifdef DISPLAY
        std::vector<glm::fvec2> StateEstimator_Impl::get_all_left_cone_points() {
            std::lock_guard<std::mutex> guard {m_mutex};        
            return m_all_left_cone_points;
        }

        std::vector<glm::fvec2> StateEstimator_Impl::get_all_right_cone_points() {
            std::lock_guard<std::mutex> guard {m_mutex};
            
            return m_all_right_cone_points;
        }

        std::vector<glm::fvec2> StateEstimator_Impl::get_left_cone_points() {
            std::lock_guard<std::mutex> guard {m_mutex};        
            return m_left_cone_points;
        }

        std::vector<glm::fvec2> StateEstimator_Impl::get_right_cone_points() {
            std::lock_guard<std::mutex> guard {m_mutex};
            
            return m_right_cone_points;
        }

        // *****REVIEW: not be needed for display
        std::vector<glm::fvec2> StateEstimator_Impl::get_raceline_points(){
            std::lock_guard<std::mutex> guard {m_mutex};
            std::stringstream ss;

            ss << "Raceline points size: " << m_raceline_points.size() << "\n";
            for (size_t i = 0; i < m_raceline_points.size(); i++)
            {
                ss << "Index: " << i << " Point x: " << m_raceline_points[i].x << "Point y: " << m_raceline_points[i].y << "\n";
            }

            return m_raceline_points;
        }


        std::vector<float> StateEstimator_Impl::get_vertices() {
            std::lock_guard<std::mutex> guard {m_mutex};

            return m_vertices;
        }

        // std::vector<GLuint> StateEstimator_Impl::get_indices() {
        //     std::lock_guard<std::mutex> guard {m_mutex};

        //     return m_indices;


        OffsetImage StateEstimator_Impl::get_offset_pixels() {
            std::lock_guard<std::mutex> guard {m_mutex};
            return m_last_offset_image;
        }
#endif

        void StateEstimator_Impl::sync_tex_info() {
            CUDA_CALL(cudaMemcpyToSymbolAsync(
                cuda_globals::curv_frame_lookup_tex_info, &m_curv_frame_lookup_tex_info, sizeof(cuda_globals::CurvFrameLookupTexInfo)
            ));
        }

        

        void StateEstimator_Impl::gen_tex_info(glm::fvec2 car_pos) {
            float xmin = car_pos.x;
            float ymin = car_pos.y;
            float xmax = car_pos.x;
            float ymax = car_pos.y;

            for (const glm::fvec2 frame : m_spline_frames) {
                xmin = std::min(xmin, frame.x);
                xmax = std::max(xmax, frame.x);
                ymin = std::min(ymin, frame.y);
                ymax = std::max(ymax, frame.y);
            }

            for (const glm::fvec2 frame : m_left_cone_points)
            {
                xmin = std::min(xmin, frame.x);
                xmax = std::max(xmax, frame.x);
                ymin = std::min(ymin, frame.y);
                ymax = std::max(ymax, frame.y);
            }

            for (const glm::fvec2 frame : m_right_cone_points)
            {
                xmin = std::min(xmin, frame.x);
                xmax = std::max(xmax, frame.x);
                ymin = std::min(ymin, frame.y);
                ymax = std::max(ymax, frame.y);
            }

            m_curv_frame_lookup_tex_info.xcenter = (xmax + xmin) / 2;
            m_curv_frame_lookup_tex_info.ycenter = (ymax + ymin) / 2;
            m_curv_frame_lookup_tex_info.width = std::max(xmax - xmin, ymax - ymin) + car_padding * 2;
        }

        void StateEstimator_Impl::render_fake_track(FakeTrackInfo &fake_track_info) {
            glBindFramebuffer(GL_FRAMEBUFFER, fake_track_info.fbo);

            if (m_follow_midline_only) {
                // ^ Replaces the texture with the render buffer (the final target)
                // Explanation: If we are only following the midline, we don't need track bounds, so we can skip the second rendering step
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_curv_frame_lookup_rbo);
            }

            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);

            // use a shader program
            glUseProgram(m_fake_track_shader_program);
            // set the relevant scale and center uniforms (constants) in the shader program
            glUniform1f(shader_scale_loc, 2.0f / m_curv_frame_lookup_tex_info.width);
            glUniform2f(shader_center_loc, m_curv_frame_lookup_tex_info.xcenter, m_curv_frame_lookup_tex_info.ycenter);

            glBindVertexArray(fake_track_info.path.vao);
            glDrawElements(GL_TRIANGLES, (fake_track_info.num_elements * 6 - 2) * 3, GL_UNSIGNED_INT, nullptr);
        }

        void StateEstimator_Impl::render_curv_frame_lookup() {
            // tells OpenGL: this is where I want to render to
            glBindFramebuffer(GL_FRAMEBUFFER, m_curv_frame_lookup_fbo);

            // set the background color, clears the depth buffer
            // (technically 2 rendering passes are done - color and depth)
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT); // TODO: remove depth buffer

            if (!no_midline_controller) {
                // use a shader program
                glUseProgram(m_curv_frame_lookup_shader_program);
                // set the relevant scale and center uniforms (constants) in the shader program
                glUniform1f(shader_scale_loc, 2.0f / m_curv_frame_lookup_tex_info.width);
                glUniform2f(shader_center_loc, m_curv_frame_lookup_tex_info.xcenter, m_curv_frame_lookup_tex_info.ycenter);

                glBindVertexArray(m_gl_path.vao);
                glBindTexture(GL_TEXTURE_2D, m_midline_fake_track.texture_color);
                glDrawArrays(GL_TRIANGLES, 0, m_num_triangles*3);
            } else {
                glUseProgram(m_double_cone_shader_program);
                glUniform1f(shader_scale_loc, 2.0f / m_curv_frame_lookup_tex_info.width);
                glUniform2f(shader_center_loc, m_curv_frame_lookup_tex_info.xcenter, m_curv_frame_lookup_tex_info.ycenter);

                glBindVertexArray(m_gl_path.vao);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, m_left_fake_track.texture_color);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, m_right_fake_track.texture_color);
                glDrawArrays(GL_TRIANGLES, 0, m_num_triangles*3);
            }
            


#ifdef DISPLAY
            if (display_on) {
                glBindFramebuffer(GL_READ_FRAMEBUFFER, m_curv_frame_lookup_fbo);
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
                glBlitFramebuffer(
                    0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width,
                    0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width,
                    GL_COLOR_BUFFER_BIT, GL_NEAREST);

                SDL_GL_SwapWindow(m_gl_window);
            }
#endif

        }

        // Whole lotta CUDA nonsense. Tread lightly.
        void StateEstimator_Impl::map_curv_frame_lookup() {
            CUDA_CALL(cudaGraphicsGLRegisterImage(&m_curv_frame_lookup_rsc, m_curv_frame_lookup_rbo, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsNone));

            if (!m_curv_frame_lookup_mapped) {
                m_curv_frame_lookup_mapped = true;

                CUDA_CALL(cudaGraphicsMapResources(1, &m_curv_frame_lookup_rsc));
            }

            cudaResourceDesc img_rsc_desc {};
            img_rsc_desc.resType = cudaResourceTypeMipmappedArray;
            CUDA_CALL(cudaGraphicsResourceGetMappedMipmappedArray(&img_rsc_desc.res.mipmap.mipmap, m_curv_frame_lookup_rsc));

            cudaTextureDesc img_tex_desc {};
            img_tex_desc.addressMode[0] = cudaAddressModeClamp;
            img_tex_desc.addressMode[1] = cudaAddressModeClamp;
            img_tex_desc.filterMode = cudaFilterModeLinear;
            img_tex_desc.readMode = cudaReadModeElementType;
            img_tex_desc.normalizedCoords = true;

            cudaTextureObject_t tex;
            CUDA_CALL(cudaCreateTextureObject(&tex, &img_rsc_desc, &img_tex_desc, nullptr));
            CUDA_CALL(cudaMemcpyToSymbolAsync(
                cuda_globals::curv_frame_lookup_tex, &tex, sizeof(cudaTextureObject_t)
            ));
        }

        /// you can't render to something when it is mapped. have to unmap before rendering
        void StateEstimator_Impl::unmap_curv_frame_lookup() {
            if (!m_curv_frame_lookup_mapped)
                return;

            CUDA_CALL(cudaGraphicsUnmapResources(1, &m_curv_frame_lookup_rsc));
            m_curv_frame_lookup_mapped = false;
        }


        // bind stuff to flags, act on flags, state machine

        /**
         * Creates the buffers to be used, as well as the descriptions of how the buffers are laid out.
         * @brief Creates the names for the vao, vbo and ebo.
         * Specifies how the vbo should be laid out, stores this in the vao.
         * Lastly, binds to the ebo.
         */
        void StateEstimator_Impl::gen_gl_path(utils::GLObj &gl_path) {
            // Generates the vao, vbo and ebo to be bound later.
            glGenVertexArrays(1, &gl_path.vao);
            glGenBuffers(1, &gl_path.vbo);
            glGenBuffers(1, &gl_path.ebo);

            glBindVertexArray(gl_path.vao);
            // OpenGL is a state machine, binding here means any relevant function call on a buffer will be on
            // m_gl_path.vbo until it is unbound.
            glBindBuffer(GL_ARRAY_BUFFER, gl_path.vbo);
            // Specifies the layout of the vertex buffer object. world_pos (2) and curv_pose (3).
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
            glEnableVertexAttribArray(1);
            // vbo unbound here, ebo bound in its place.
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl_path.ebo);

            // vao is unbound, saving the settings.
            glBindVertexArray(0);
        }

        // Track bounds version
        void StateEstimator_Impl::fill_path_buffers_cones(){
            const size_t num_left_cones = m_left_cone_points.size();
            const size_t num_right_cones = m_right_cone_points.size();
            m_num_triangles = 0;
            std::vector<StateEstimator_Impl::Vertex> indices;
            std::vector<StateEstimator_Impl::Vertex> vertices;
            //vertices.reserve(num_left_cones + num_right_cones);
            for (size_t i = 0; i < num_left_cones; ++i) {
                    glm::fvec2 l1 = m_left_cone_points.at(i);
                    vertices.push_back({{l1.x, l1.y}, {0.0f, 0.3f, 0.0f}});
            }
            for (size_t i = 0; i < num_right_cones; ++i) {
                    glm::fvec2 r1 = m_right_cone_points.at(i);
                    vertices.push_back({{r1.x, r1.y}, {0.0f, 0.3f, 0.0f}});
            }
            float distance2;
            std::vector<GLuint> temp;
            for(size_t i = 0; i < num_left_cones; ++i){
                glm::fvec2 l1 = m_left_cone_points.at(i);
                distance2 = 0;
                temp.clear();
                for(size_t j = 0; j < num_right_cones; ++j){
                    glm::fvec2 r1 = m_right_cone_points.at(j);
                    distance2 = (l1.x - r1.x)*(l1.x - r1.x) + (l1.y - r1.y)*(l1.y - r1.y);
                    if(distance2 < triangle_threshold_squared)
                    {
                        temp.push_back(j);
                    }
                }
                if(temp.size() > 1){
                    for(size_t k = 0; k < temp.size()-1; ++k){
                        indices.push_back(vertices.at(i));
                        indices.push_back(vertices.at(temp.at(k)+ num_left_cones));
                        indices.push_back(vertices.at(temp.at(k+1) + num_left_cones));
                        m_num_triangles += 1;
                    }
                }
            }
            for(size_t i = 0; i < num_right_cones; ++i){
                glm::fvec2 r1 = m_right_cone_points.at(i);
                distance2 = 0;
                temp.clear();
                for(size_t j = 0; j < num_left_cones; j++){
                    glm::fvec2 l1 = m_left_cone_points.at(j);
                    distance2 = (l1.x - r1.x)*(l1.x - r1.x) + (l1.y - r1.y)*(l1.y - r1.y);
                    if(distance2 < triangle_threshold_squared)
                    {
                        temp.push_back(j);
                    }
                }
                if(temp.size() > 1){
                    for(size_t k = 0; k < temp.size()-1; ++k){
                        indices.push_back(vertices.at(i + num_left_cones));
                        indices.push_back(vertices.at(temp.at(k)));
                        indices.push_back(vertices.at(temp.at(k+1)));
                        m_num_triangles += 1;
                    }
                }

            }
            // TODO: decide whether we are going to keep using glDrawArrays or try to fix glDrawElements

            std::stringstream ss;
            ss << "Start of right at: " << num_left_cones;
            for(size_t i = 0; i < indices.size(); i++){
                ss << "Index: " << i << " Point x: " << indices.at(i).world.x << "Point y: "<< indices.at(i).world.y << "\n";
            }
            // if(indices.size() > 2) {
            //     for(size_t i = 0; i < indices.size()-2; i += 3){
            //         ss << "Index: " << indices.at(i) << " 2: " << indices.at(i+1) << " 3: " << indices.at(i+2) <<"------";
            //     }
            // }
    
            RCLCPP_DEBUG(m_logger_obj, ss.str().c_str());
            glBindBuffer(GL_ARRAY_BUFFER, m_gl_path.vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(StateEstimator_Impl::Vertex) * indices.size(), indices.data(), GL_DYNAMIC_DRAW);

            // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gl_path.ebo);
            // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_DYNAMIC_DRAW);

        }
        
        // Offset from spline version
        /**
         * Given spline frames, fills a framebuffer object with the fake track corresponding to the spline. Used for track progress.
         * The framebuffer object will be used as a texture object to be sampled from the state estimator.
         */
        void StateEstimator_Impl::fill_path_buffers_for_fake_track(FakeTrackInfo &fake_track_info, const std::vector<glm::fvec2>& base_points) {
            for (const auto& frame : base_points) {
                paranoid_assert(!isnan_vec(frame));
            }
            struct Vertex {
                struct {
                    float x;
                    float y;
                } world;

                struct {
                    float progress;
                    float offset;
                    float heading;
                } curv;
            };

            const float radius = fake_track_width * 0.5f;
            const size_t n = base_points.size();

            if (n < 2) {
                throw ControllerError("less than 2 spline frames! (bruh saket and/or saket)");
            }

            std::vector<Vertex> vertices;
            std::vector<GLuint> indices;

            float total_progress = 0;
            for (size_t i = 0; i < n - 1; i++) {
                glm::fvec2 p1 = base_points[i];
                glm::fvec2 p2 = base_points[i + 1];

                glm::fvec2 unit_vec = glm::length(p2 - p1) != 0 ? glm::normalize(p2 - p1) : glm::fvec2(0, 0);
                // This creates a longitudinal buffer at the start and the end of the spline for the fake track to be rendered
                if (i == 0) {
                    p1 = p1 - unit_vec * car_padding;
                } else if (i == n - 2) {
                    p2 = p2 + unit_vec * car_padding;
                }

                glm::fvec2 disp = p2 - p1;
                float new_progress = glm::length(disp); // TODO: figure out a way to normalize without some arbitrary magic number
                paranoid_assert(!isnan(new_progress));
                // 1. go through the vector and divide based on total progress
                // 2. set total progress to be a member variable, then use that as a uniform, thus passing it into the fragment shader
                float segment_heading = std::atan2(disp.y, disp.x);


                glm::fvec2 prev = i == 0 ? p1 : base_points[i - 1];
                float secant_heading = std::atan2(p2.y - prev.y, p2.x - prev.x);

                glm::fvec2 dir = glm::normalize(disp);
                glm::fvec2 normal = glm::fvec2(-dir.y, dir.x);

                glm::fvec2 low1 = p1 - normal * radius;
                glm::fvec2 low2 = p2 - normal * radius;
                glm::fvec2 high1 = p1 + normal * radius;
                glm::fvec2 high2 = p2 + normal * radius;

                if (i == 0)
                {
                    vertices.push_back({{p1.x, p1.y}, {total_progress, 0.0f, 0.0f}});
                }
                vertices.push_back({{p2.x, p2.y}, {total_progress + new_progress, 0.0f, 0.0f}});

                // I set offset to be 1.0 to prevent plateauing
                vertices.push_back({{low1.x, low1.y}, {total_progress, radius, 0.0f}});
                vertices.push_back({{low2.x, low2.y}, {total_progress + new_progress, radius, 0.0f}});
                vertices.push_back({{high1.x, high1.y}, {total_progress, radius, 1.0f}});
                vertices.push_back({{high2.x, high2.y}, {total_progress + new_progress, radius, 1.0f}});

                const GLuint p1i = i == 0 ? 0 : (i - 1) * 5 + 1;
                const GLuint p2i = i * 5 + 1;
                const GLuint l1i = i * 5 + 2;
                const GLuint l2i = i * 5 + 3;
                const GLuint h1i = i * 5 + 4;
                const GLuint h2i = i * 5 + 5;

                indices.push_back(p1i);
                indices.push_back(p2i);
                indices.push_back(h2i);

                indices.push_back(h1i);
                indices.push_back(p1i);
                indices.push_back(h2i);

                indices.push_back(l1i);
                indices.push_back(l2i);
                indices.push_back(p2i);

                indices.push_back(p1i);
                indices.push_back(l1i);
                indices.push_back(p2i);

                if (i > 0) {
                    const GLuint lpi = (i - 1) * 5 + 3;
                    const GLuint hpi = (i - 1) * 5 + 5;

                    indices.push_back(hpi);
                    indices.push_back(p1i);
                    indices.push_back(h1i);

                    indices.push_back(lpi);
                    indices.push_back(l1i);
                    indices.push_back(p1i);
                }

                total_progress += new_progress;
            }

            glBindBuffer(GL_ARRAY_BUFFER, fake_track_info.path.vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fake_track_info.path.ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_DYNAMIC_DRAW);

            fake_track_info.num_elements = base_points.size();
        }
    }
}

