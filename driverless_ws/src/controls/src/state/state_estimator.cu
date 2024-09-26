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
#include <glm/common.hpp>
#include <mppi/functors.cuh>
#include <SDL2/SDL_video.h>


namespace controls {
    namespace state {

        void StateProjector::print_history() const {
            std::cout << "---- BEGIN HISTORY ---\n";
            for (const Record& record : m_history_since_pose) {
                switch (record.type) {
                    case Record::Type::Action:
                        std::cout << "Action: " << record.action[0] << ", " << record.action[1] << std::endl;
                        break;

                    case Record::Type::Speed:
                        std::cout << "Speed: " << record.speed << std::endl;
                        break;

                    case Record::Type::Pose:
                        std::cout << "Pose: " << record.pose.x << ", " << record.pose.y << ", " << record.pose.yaw << std::endl;
                        break;

                    default:
                        throw new std::runtime_error("bruh. invalid record type bruh. (in print history)");
                }
            }
            std::cout << "---END HISTORY---" << std::endl;
        }

        void StateProjector::record_action(Action action, rclcpp::Time time) {
            // std::cout << "Recording action " << action[0] << ", " << action[1] << " at time " << time.nanoseconds() << std::endl;

            // change to assert(!m_pose_record.has_value() || time >= m_pose_record.value().time)
            assert(m_pose_record.has_value() && time >= m_pose_record.value().time
                && "call me marty mcfly the way im time traveling");

            m_history_since_pose.insert(Record {
                .action = action,
                .time = time,
                .type = Record::Type::Action
            });

            // print_history();
        }

        void StateProjector::record_speed(float speed, rclcpp::Time time) {
            // std::cout << "Recording speed " << speed << " at time " << time.nanoseconds() << std::endl;

            if (m_pose_record.has_value() && time < m_pose_record.value().time) {
                if (time > m_init_speed.time) {
                    m_init_speed = Record {
                        .speed = speed,
                        .time = time,
                        .type = Record::Type::Speed
                    };
                }
            } else {
                m_history_since_pose.insert(Record {
                    .speed = speed,
                    .time = time,
                    .type = Record::Type::Speed
                });
            }

            // print_history();
        }

        void StateProjector::record_pose(float x, float y, float yaw, rclcpp::Time time) {
            // std::cout << "Recording pose " << x << ", " << y << ", " << yaw << " at time " << time.nanoseconds() << std::endl;

            m_pose_record = Record {
                .pose = {
                    .x = x,
                    .y = y,
                    .yaw = yaw
                },
                .time = time,
                .type = Record::Type::Pose
            };

            auto record_iter = m_history_since_pose.begin();
            for (; record_iter != m_history_since_pose.end(); ++record_iter) {
                if (record_iter->time > time) {
                    break;
                }

                switch (record_iter->type) {
                    case Record::Type::Action:
                        m_init_action = *record_iter;
                        break;

                    case Record::Type::Speed:
                        m_init_speed = *record_iter;
                        break;

                    default:
                        throw new std::runtime_error("bruh. invalid record type bruh. (in record pose)");
                }
            }

            m_history_since_pose.erase(m_history_since_pose.begin(), record_iter);

            // print_history();
        }

        State StateProjector::project(const rclcpp::Time& time, LoggerFunc logger) const {
            assert(m_pose_record.has_value() && "State projector has not recieved first pose");
            // std::cout << "Projecting to " << time.nanoseconds() << std::endl;

            State state;
            state[state_x_idx] = m_pose_record.value().pose.x;
            state[state_y_idx] = m_pose_record.value().pose.y;
            state[state_yaw_idx] = m_pose_record.value().pose.yaw;
            state[state_speed_idx] = m_init_speed.speed;

            const auto first_time = m_history_since_pose.empty() ? time : m_history_since_pose.begin()->time;
            const float delta_time = (first_time.nanoseconds() - m_pose_record.value().time.nanoseconds()) / 1e9f;
            // std::cout << "delta time: " << delta_time << std::endl;
            assert(delta_time > 0 && "RUH ROH. Delta time for propogation delay simulation was negative.   : (");
            // simulates up to first_time
            ONLINE_DYNAMICS_FUNC(state.data(), m_init_action.action.data(), state.data(), delta_time);

            rclcpp::Time sim_time = first_time;
            Action last_action = m_init_action.action;
            for (auto record_iter = m_history_since_pose.begin(); record_iter != m_history_since_pose.end(); ++record_iter) {
                // checks if we're on last record
                const auto next_time = std::next(record_iter) == m_history_since_pose.end() ? time : std::next(record_iter)->time;

                const float delta_time = (next_time - sim_time).nanoseconds() / 1e9f;
                assert(delta_time >= 0 && "RUH ROH. Delta time for propogation delay simulation was negative.   : (");

                switch (record_iter->type) {
                    case Record::Type::Action:
                        ONLINE_DYNAMICS_FUNC(state.data(), record_iter->action.data(), state.data(), delta_time);
                        last_action = record_iter->action;
                        break;

                    case Record::Type::Speed:
                        char logger_buf[70];
                        snprintf(logger_buf, 70, "Predicted speed: %f\nActual speed: %f", state[state_speed_idx], record_iter->speed);
                        //std::cout << logger_buf << std::endl;
                        state[state_speed_idx] = record_iter->speed;
                        ONLINE_DYNAMICS_FUNC(state.data(), last_action.data(), state.data(), delta_time);
                        break;

                    default:
                        throw new std::runtime_error("bruh. invalid record type bruh. (in simulation)");
                }

                sim_time = next_time;
            }

            return state;
        }

        bool StateProjector::is_ready() const {
            return m_pose_record.has_value();
        }


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
        constexpr const char* vertex_source = R"(
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
        constexpr const char* fragment_source = R"(
            #version 330 core

            in vec3 o_curv_pose;

            out vec4 FragColor;

            void main() {
                FragColor = vec4(o_curv_pose, 1.0f);
            }
        )";



        // methods

        StateEstimator_Impl::StateEstimator_Impl(std::mutex& mutex, LoggerFunc logger)
            : m_mutex {mutex}, m_logger {logger} {
            std::lock_guard<std::mutex> guard {mutex};

            m_logger("initializing state estimator");
#ifdef DISPLAY
            m_gl_window = utils::create_sdl2_gl_window(
                "Spline Frame Lookup", curv_frame_lookup_tex_width, curv_frame_lookup_tex_width,
                0, &m_gl_context
            );
#else
            // dummy window to create opengl context for curv frame buffer
            m_gl_window = utils::create_sdl2_gl_window(
                "Spline Frame Lookup Dummy", 1, 1,
                SDL_WINDOW_HIDDEN, &m_gl_context
            );
#endif

            m_logger("making state estimator gl context current");
            utils::make_gl_current_or_except(m_gl_window, m_gl_context);

            m_logger("compiling state estimator shaders");
            m_gl_path_shader = utils::compile_shader(vertex_source, fragment_source);

            m_logger("setting state estimator gl properties");
            glClearColor(0.0f, 0.0f, 0.0f, -1.0f);

            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);

            glViewport(0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width);

            m_logger("generating state estimator gl buffers");
            gen_curv_frame_lookup_framebuffer();
            gen_gl_path();

            glFinish();
            utils::make_gl_current_or_except(m_gl_window, nullptr);
            m_logger("finished state estimator initialization");
        }

        void StateEstimator_Impl::gen_curv_frame_lookup_framebuffer() {
            glGenFramebuffers(1, &m_curv_frame_lookup_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, m_curv_frame_lookup_fbo);

            glGenRenderbuffers(1, &m_curv_frame_lookup_rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, m_curv_frame_lookup_rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_curv_frame_lookup_rbo);

            GLuint depth_rbo;
            glGenRenderbuffers(1, &depth_rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, curv_frame_lookup_tex_width,  curv_frame_lookup_tex_width);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                throw std::runtime_error("Framebuffer is not complete");
            }
        }

        StateEstimator_Impl::~StateEstimator_Impl() {
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }

        void StateEstimator_Impl::on_spline(const SplineMsg& spline_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};

            m_logger("beginning state estimator spline processing");

            m_spline_frames.clear();
            m_spline_frames.reserve(spline_msg.frames.size());

            for (const auto& frame : spline_msg.frames) {
                m_spline_frames.push_back({
                    static_cast<float>(frame.y),
                    static_cast<float>(frame.x)
                });
            }

            if constexpr (reset_pose_on_spline) {
                m_state_projector.record_pose(0, 0, 0, spline_msg.orig_data_stamp);
            }

            m_orig_spline_data_stamp = spline_msg.orig_data_stamp;

            m_logger("finished state estimator spline processing");
        }
        

        void StateEstimator_Impl::on_cone(const ConeMsg& cone_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};

            m_logger("beginning state estimator cone processing");
    
            m_left_cone_positions.clear();
            m_left_cone_positions.reserve(cone_msg.blue_cones.size());
            m_right_cone_positions.clear();
            m_right_cone_positions.reserve(cone_msg.yellow_cones.size());

            for (const auto& cone_point : cone_msg.blue_cones) {
                // TODO when perceptions gets their shit together
                float cone_y = static_cast<float>(cone_point.y);
                float cone_x = static_cast<float>(cone_point.x);
                float distance = cone_y * cone_y + cone_x * cone_x;
                m_left_cone_positions.push_back(
                    std::make_pair(
                        distance,
                        glm::fvec2(cone_y, cone_x)
                    )
                );
            }

            for (const auto& cone_point : cone_msg.yellow_cones) {
                float cone_y = static_cast<float>(cone_point.y);
                float cone_x = static_cast<float>(cone_point.x);
                float distance = cone_y * cone_y + cone_x * cone_x;
                m_right_cone_positions.push_back(
                    std::make_pair(
                        distance,
                        glm::fvec2(cone_y,cone_x) 
                    )
                    // TODO when perceptions gets their shit together
                );
            }

            if constexpr (reset_pose_on_spline) {
                m_state_projector.record_pose(0, 0, 0, cone_msg.orig_data_stamp);
            }
            
            m_logger("finished state estimator cone processing");
        }

        void StateEstimator_Impl::on_twist(const TwistMsg &twist_msg, const rclcpp::Time &time) {
            // TODO: whats up with all these mutexes
            std::lock_guard<std::mutex> guard {m_mutex};

            const float speed = std::sqrt(
                twist_msg.twist.linear.x * twist_msg.twist.linear.x
                + twist_msg.twist.linear.y * twist_msg.twist.linear.y);

            m_state_projector.record_speed(speed, time);
        }

        void StateEstimator_Impl::on_pose(const PoseMsg &pose_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};

            m_state_projector.record_pose(
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.orientation.z,
                pose_msg.header.stamp);
        }

        rclcpp::Time StateEstimator_Impl::get_orig_spline_data_stamp() {
            std::lock_guard<std::mutex> guard {m_mutex};

            return m_orig_spline_data_stamp;
        }

        void StateEstimator_Impl::record_control_action(const Action& action, const rclcpp::Time& time) {
            std::lock_guard<std::mutex> guard {m_mutex};

            // record actions in the future (when they are actually requested by the actuator)
            m_state_projector.record_action(action, rclcpp::Time {
                    time.nanoseconds()
                    + static_cast<int64_t>(approx_propogation_delay * 1e9f),
                    default_clock_type
                }
            );
        }

        void StateEstimator_Impl::sync_to_device(const rclcpp::Time& time) {
            std::lock_guard<std::mutex> guard {m_mutex};

            m_logger("beginning state estimator device sync");

            m_logger("projecting current state");
            const State state = m_state_projector.project(
                rclcpp::Time {
                    time.nanoseconds()
                    + static_cast<int64_t>((approx_propogation_delay + approx_mppi_time) * 1e9f),
                    default_clock_type
                }, m_logger
            );

            // enable openGL
            utils::make_gl_current_or_except(m_gl_window, m_gl_context);

            m_logger("generating spline frame lookup texture info...");

            gen_tex_info({state[state_x_idx], state[state_y_idx]});

            m_logger("filling OpenGL buffers...");
            // takes car position, places them in the vertices
            fill_path_buffers({state[state_x_idx], state[state_y_idx]});

            m_logger("unmapping CUDA curv frame lookup texture for OpenGL rendering");
            unmap_curv_frame_lookup();

            // render the lookup table
            m_logger("rendering curv frame lookup table...");
            render_curv_frame_lookup();

            m_logger("mapping OpenGL curv frame texture back to CUDA");
            map_curv_frame_lookup();

            m_logger("syncing world state to device");
            m_synced_projected_state = state;
            sync_world_state();

            m_logger("syncing spline frame lookup texture info to device");
            sync_tex_info();

            utils::sync_gl_and_unbind_context(m_gl_window);
            m_logger("finished state estimator device sync");
        }

        bool StateEstimator_Impl::is_ready() {
            std::lock_guard<std::mutex> guard {m_mutex};

            return m_state_projector.is_ready();
        }

        State StateEstimator_Impl::get_projected_state() {
            std::lock_guard<std::mutex> guard {m_mutex};

            return m_synced_projected_state;
        }

        void StateEstimator_Impl::set_logger(LoggerFunc logger) {
            std::lock_guard<std::mutex> guard {m_mutex};

            m_logger = logger;
        }

#ifdef DISPLAY
        std::vector<glm::fvec2> StateEstimator_Impl::get_spline_frames() {
            std::lock_guard<std::mutex> guard {m_mutex};

            std::vector<glm::fvec2> res (m_spline_frames.size());
            for (size_t i = 0; i < m_spline_frames.size(); i++) {
                res[i] = {m_spline_frames[i].x, m_spline_frames[i].y};
            }
            return res;
        }

        std::vector<glm::fvec2> StateEstimator_Impl::get_left_cone_frames() {
            std::lock_guard<std::mutex> guard {m_mutex};

            std::vector<glm::fvec2> res (m_left_cone_positions.size());
            for (size_t i = 0; i < m_left_cone_positions.size(); i++) {
                res[i] = {m_left_cone_positions.at(i).second.x, m_left_cone_positions.at(i).second.y};
            }
            return res;
        }

        std::vector<glm::fvec2> StateEstimator_Impl::get_right_cone_frames() {
            std::lock_guard<std::mutex> guard {m_mutex};

            std::vector<glm::fvec2> res (m_right_cone_positions.size());
            for (size_t i = 0; i < m_right_cone_positions.size(); i++) {
                res[i] = {m_right_cone_positions.at(i).second.x, m_right_cone_positions.at(i).second.y};
            }
            return res;
        }



        void StateEstimator_Impl::get_offset_pixels(OffsetImage &offset_image) {
            std::lock_guard<std::mutex> guard {m_mutex};

            SDL_GLContext prev_context = SDL_GL_GetCurrentContext();
            SDL_Window* prev_window = SDL_GL_GetCurrentWindow();

            utils::make_gl_current_or_except(m_gl_window, m_gl_context);

            offset_image.pixels = std::vector<float>(curv_frame_lookup_tex_width * curv_frame_lookup_tex_width);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, m_curv_frame_lookup_fbo);
            glReadPixels(
                0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width,
                GL_GREEN, GL_FLOAT,
                offset_image.pixels.data()
            );

            offset_image.pix_width = curv_frame_lookup_tex_width;
            offset_image.pix_height = curv_frame_lookup_tex_width;
            offset_image.center = {m_curv_frame_lookup_tex_info.xcenter, m_curv_frame_lookup_tex_info.ycenter};
            offset_image.world_width = m_curv_frame_lookup_tex_info.width;

            utils::sync_gl_and_unbind_context(m_gl_window);
            utils::make_gl_current_or_except(prev_window, prev_context);
        }
#endif

        void StateEstimator_Impl::sync_world_state() {
            CUDA_CALL(cudaMemcpyToSymbolAsync(
                cuda_globals::curr_state, m_synced_projected_state.data(), state_dims * sizeof(float)
            ));
        }

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

            m_curv_frame_lookup_tex_info.xcenter = (xmax + xmin) / 2;
            m_curv_frame_lookup_tex_info.ycenter = (ymax + ymin) / 2;
            m_curv_frame_lookup_tex_info.width = std::max(xmax - xmin, ymax - ymin) + car_padding * 2;
        }

        void StateEstimator_Impl::render_curv_frame_lookup() {
            // tells OpenGL: this is where I want to render to
            glBindFramebuffer(GL_FRAMEBUFFER, m_curv_frame_lookup_fbo);

            // set the background color, clears the depth buffer
            // (technically 2 rendering passes are done - color and depth)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // use a shader program
            glUseProgram(m_gl_path_shader);
            // set the relevant scale and center uniforms (constants) in the shader program
            glUniform1f(shader_scale_loc, 2.0f / m_curv_frame_lookup_tex_info.width);
            glUniform2f(shader_center_loc, m_curv_frame_lookup_tex_info.xcenter, m_curv_frame_lookup_tex_info.ycenter);

            glBindVertexArray(m_gl_path.vao);
            // relies on the element buffer object already being bound
            glDrawElements(GL_TRIANGLES, m_num_triangles, GL_UNSIGNED_INT, nullptr);

#ifdef DISPLAY
            glBindFramebuffer(GL_READ_FRAMEBUFFER, m_curv_frame_lookup_fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glBlitFramebuffer(
                0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width,
                0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width,
                GL_COLOR_BUFFER_BIT, GL_NEAREST
            );

            SDL_GL_SwapWindow(m_gl_window);
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
        void StateEstimator_Impl::gen_gl_path() {
            // Generates the names for the vao, vbo and ebo to be referenced later, such as to bind.
            glGenVertexArrays(1, &m_gl_path.vao);
            glGenBuffers(1, &m_gl_path.vbo);
            glGenBuffers(1, &m_gl_path.ebo);

            glBindVertexArray(m_gl_path.vao);
            // OpenGL is a state machine, binding here means any relevant function call on a buffer will be on
            // m_gl_path.vbo until it is unbound.
            glBindBuffer(GL_ARRAY_BUFFER, m_gl_path.vbo);
            // Specifies the layout of the vertex buffer object. world_pos (2) and curv_pose (3).
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
            glEnableVertexAttribArray(1);
            // vbo unbound here, ebo bound in its place.
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gl_path.ebo);

            // vao is unbound.
            glBindVertexArray(0);
        }

        // THIS IS WHERE I EDIT
        void StateEstimator_Impl::fill_path_buffers(glm::fvec2 car_pos) {
            struct Vertex {
                struct {
                    float x;
                    float y;
                } world;

                /// Curvilinear coordinates. Progress = distance along spline, offset = perpendicular distance
                /// from spline, heading = angle relative to spline.
                struct {
                    float progress;
                    float offset;
                    float heading;
                } curv;
            };

            const size_t num_splines = m_spline_frames.size();
            const size_t num_left_cones = m_left_cone_positions.size();
            const size_t num_right_cones = m_right_cone_positions.size();

            std::stringstream ss;
            ss << "# splines: " << num_splines << "# Left cones: " << num_left_cones << "# Right cones: " << num_right_cones << "\n";
            RCLCPP_WARN(m_logger, ss.str().c_str());

            std::vector<Vertex> vertices;
            std::vector<GLuint> indices;

            auto cmp = [](const std::pair<float, glm::fvec2>& a, const std::pair<float, glm::fvec2>& b) {
                return a.first < b.first;
            };

            std::sort(m_left_cone_positions.begin(), m_left_cone_positions.end(), cmp);
            std::sort(m_right_cone_positions.begin(), m_right_cone_positions.end(), cmp);

            m_num_triangles = 0;
            for (size_t i = 0; i < std::min(num_left_cones, num_right_cones) - 1; ++i) {
                std::cout << "BEGIN" << std::endl;
                glm::fvec2 l1 = m_left_cone_positions.at(i).second;
                glm::fvec2 l2 = m_left_cone_positions.at(i + 1).second;
                glm::fvec2 r1 = m_right_cone_positions.at(i).second;
                glm::fvec2 r2 = m_right_cone_positions.at(i + 1).second;
                std::cout << "END" << std::endl;

                vertices.push_back({{l1.x, l1.y}, {10.0f, 0.0f, 1.0f}});
                vertices.push_back({{l2.x, l2.y}, {10.0f, 0.0f, 1.0f}});
                vertices.push_back({{r1.x, r1.y}, {10.0f, 0.0f, 1.0f}});
                vertices.push_back({{r2.x, r2.y}, {10.0f, 0.0f, 1.0f}});

                const GLuint l1i = i * 4;
                const GLuint l2i = i * 4 + 1;
                const GLuint r1i = i * 4 + 2;
                const GLuint r2i = i * 4 + 3;

                indices.push_back(l1i);
                indices.push_back(l2i);
                indices.push_back(r1i);

                indices.push_back(r1i);
                indices.push_back(r2i);
                indices.push_back(l2i);

                m_num_triangles += 2;

            }

            glBindBuffer(GL_ARRAY_BUFFER, m_gl_path.vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gl_path.ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_DYNAMIC_DRAW);
        }
    }
}

