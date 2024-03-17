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

#include <SDL2/SDL_video.h>


namespace controls {
    namespace state {

        std::shared_ptr<StateEstimator> StateEstimator::create(std::mutex& mutex) {
            return std::make_shared<StateEstimator_Impl>(mutex);
        }


        // StateEstimator_Impl helpers

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

        constexpr const char* fragment_source = R"(
            #version 330 core

            in vec3 o_curv_pose;

            out vec4 FragColor;

            void main() {
                FragColor = vec4(o_curv_pose, 1.0f);
            }
        )";

        // methods

        StateEstimator_Impl::StateEstimator_Impl(std::mutex& mutex)
            : m_mutex {mutex}, m_curv_frame_lookup_mapped {false} {
            std::lock_guard<std::mutex> guard {mutex};

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

            utils::make_gl_current_or_except(m_gl_window, m_gl_context);

            m_gl_path_shader = utils::compile_shader(vertex_source, fragment_source);

            glClearColor(0.0f, 0.0f, 0.0f, -1.0f);

            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);

            glViewport(0, 0, curv_frame_lookup_tex_width, curv_frame_lookup_tex_width);

            gen_curv_frame_lookup_framebuffer();
            gen_gl_path();

            glFinish();
            utils::make_gl_current_or_except(m_gl_window, nullptr);
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

            std::cout << "------- ON SPLINE -----" << std::endl;

            utils::make_gl_current_or_except(m_gl_window, m_gl_context);

            m_spline_frames.clear();
            m_spline_frames.reserve(spline_msg.frames.size());

            for (const auto& frame : spline_msg.frames) {
                m_spline_frames.push_back({
                    static_cast<float>(frame.x),
                    static_cast<float>(frame.y)
                });
            }

            std::cout << "generating spline frame lookup texture info..." << std::endl;
            gen_tex_info({m_world_state[state_x_idx], m_world_state[state_y_idx]});
            std::cout << "xcenter: " << m_curv_frame_lookup_tex_info.xcenter
                      << " ycenter: " << m_curv_frame_lookup_tex_info.ycenter <<
                         " width: " << m_curv_frame_lookup_tex_info.width
            << std::endl;

            std::cout << "filling OpenGL buffers..." << std::endl;
            fill_path_buffers({m_world_state[state_x_idx], m_world_state[state_y_idx]});

            utils::sync_gl_and_unbind_context(m_gl_window);

            m_spline_ready = true;

            std::cout << "-------------------\n" << std::endl;
        }

        void StateEstimator_Impl::on_world_twist(const TwistMsg &twist_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};

            const float yaw = m_world_state[state_yaw_idx];
            const float car_xdot = twist_msg.twist.linear.x * std::cos(yaw) + twist_msg.twist.linear.y * std::sin(yaw);
            const float car_ydot = -twist_msg.twist.linear.x * std::sin(yaw) + twist_msg.twist.linear.y * std::cos(yaw);
            const float car_yawdot = twist_msg.twist.angular.z;

            m_world_state[state_car_xdot_idx] = car_xdot;
            m_world_state[state_car_ydot_idx] = car_ydot;
            m_world_state[state_yawdot_idx] = car_yawdot;

            m_world_twist_ready = true;
        }

        void StateEstimator_Impl::on_world_quat(const QuatMsg &quat_msg) {
            using namespace glm;
            std::lock_guard<std::mutex> guard {m_mutex};

            const fquat quat = dquat(
                quat_msg.quaternion.w, quat_msg.quaternion.x, quat_msg.quaternion.y, quat_msg.quaternion.z
            );

            const fmat3x3 rot = mat3_cast(quat);
            const fvec3 ihatprime = rot * fvec3(1, 0, 0);
            const float yaw = std::atan2(ihatprime.y, ihatprime.x);

            m_world_state[state_yaw_idx] = yaw;

            m_world_yaw_ready = true;
        }

        // const float w = quat_msg.quaternion.w;
        // const float x = quat_msg.quaternion.x;
        // const float y = quat_msg.quaternion.y;
        // const float z = quat_msg.quaternion.z;

        // const float yaw = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));

        void StateEstimator_Impl::on_world_pose(const PoseMsg &pose_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};

            m_world_state[state_x_idx] = pose_msg.pose.position.x;
            m_world_state[state_y_idx] = pose_msg.pose.position.y;
            m_world_state[state_yaw_idx] = pose_msg.pose.orientation.z;

            m_world_yaw_ready = true;
        }

        void StateEstimator_Impl::on_state(const StateMsg& state_msg) {
            std::lock_guard<std::mutex> guard {m_mutex};

            std::cout << "------- ON STATE -----" << std::endl;

            m_world_state[state_x_idx] = state_msg.x;
            m_world_state[state_y_idx] = state_msg.y;
            m_world_state[state_yaw_idx] = state_msg.yaw;
            m_world_state[state_car_xdot_idx] = state_msg.xcar_dot;
            m_world_state[state_car_ydot_idx] = state_msg.ycar_dot;
            m_world_state[state_yawdot_idx] = state_msg.yaw_dot;
            m_world_state[state_my_idx] = state_msg.moment_y;
            m_world_state[state_fz_idx] = state_msg.downforce;
            m_world_state[state_whl_speed_f_idx] = state_msg.whl_speed_f;
            m_world_state[state_whl_speed_r_idx] = state_msg.whl_speed_r;

            m_world_twist_ready = true;
            m_world_yaw_ready = true;

            std::cout << "-------------------\n" << std::endl;
        }

        void StateEstimator_Impl::sync_to_device(float swangle) {
            std::lock_guard<std::mutex> guard {m_mutex};

            std::cout << "Publishing state" << std::endl;
            for (float dim : m_world_state)
            {
                std::cout << dim << " ";
            }

            // TODO: make wheel speed estimation optional
            estimate_whl_speeds(swangle);

            utils::make_gl_current_or_except(m_gl_window, m_gl_context);

            std::cout << "unmapping CUDA curv frame lookup texture for OpenGL rendering ..." << std::endl;
            unmap_curv_frame_lookup();

            std::cout << "rendering curv frame lookup table..." << std::endl;
            render_curv_frame_lookup();

            std::cout << "mapping OpenGL curv frame texture back to CUDA..." << std::endl;
            map_curv_frame_lookup();

            std::cout << "syncing world state to device..." << std::endl;
            sync_world_state();

            std::cout << "syncing spline frame lookup texture info to device..." << std::endl;
            sync_tex_info();

            utils::sync_gl_and_unbind_context(m_gl_window);
        }

        bool StateEstimator_Impl::is_ready() {
            return m_spline_ready && m_world_twist_ready && m_world_yaw_ready;
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
                cuda_globals::curr_state, &m_world_state, state_dims * sizeof(float)
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
            m_curv_frame_lookup_tex_info.width = std::max(xmax - xmin, ymax - ymin) + curv_frame_lookup_padding * 2;
        }

        void StateEstimator_Impl::render_curv_frame_lookup() {
            glBindFramebuffer(GL_FRAMEBUFFER, m_curv_frame_lookup_fbo);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glUseProgram(m_gl_path_shader);
            glUniform1f(shader_scale_loc, 2.0f / m_curv_frame_lookup_tex_info.width);
            glUniform2f(shader_center_loc, m_curv_frame_lookup_tex_info.xcenter, m_curv_frame_lookup_tex_info.ycenter);

            glBindVertexArray(m_gl_path.vao);
            glDrawElements(GL_TRIANGLES, (m_spline_frames.size() * 6 - 2) * 3, GL_UNSIGNED_INT, nullptr);

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

        void StateEstimator_Impl::unmap_curv_frame_lookup() {
            if (!m_curv_frame_lookup_mapped)
                return;

            CUDA_CALL(cudaGraphicsUnmapResources(1, &m_curv_frame_lookup_rsc));
            m_curv_frame_lookup_mapped = false;
        }

        void StateEstimator_Impl::gen_gl_path() {
            glGenVertexArrays(1, &m_gl_path.vao);
            glGenBuffers(1, &m_gl_path.vbo);
            glGenBuffers(1, &m_gl_path.ebo);

            glBindVertexArray(m_gl_path.vao);
            glBindBuffer(GL_ARRAY_BUFFER, m_gl_path.vbo);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gl_path.ebo);

            glBindVertexArray(0);
        }

        void StateEstimator_Impl::fill_path_buffers(glm::fvec2 car_pos) {
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

            const float radius = track_width * 0.5f;
            const size_t n = m_spline_frames.size();

            std::vector<Vertex> vertices;
            std::vector<GLuint> indices;

            float total_progress = 0;
            for (size_t i = 0; i < n - 1; i++) {
                glm::fvec2 p1 = m_spline_frames[i];
                glm::fvec2 p2 = m_spline_frames[i + 1];

                glm::fvec2 disp = p2 - p1;
                float new_progress = glm::length(disp);
                float segment_heading = std::atan2(disp.y, disp.x);


                glm::fvec2 prev = i == 0 ? p1 : m_spline_frames[i - 1];
                float secant_heading = std::atan2(p2.y - prev.y, p2.x - prev.x);

                glm::fvec2 dir = glm::normalize(disp);
                glm::fvec2 normal = glm::fvec2(-dir.y, dir.x);

                glm::fvec2 low1 = p1 - normal * radius;
                glm::fvec2 low2 = p2 - normal * radius;
                glm::fvec2 high1 = p1 + normal * radius;
                glm::fvec2 high2 = p2 + normal * radius;

                if (i == 0) {
                    vertices.push_back({{p1.x, p1.y}, {total_progress, 0.0f, segment_heading}});
                }
                vertices.push_back({{p2.x, p2.y}, {total_progress + new_progress, 0.0f, secant_heading}});

                vertices.push_back({{low1.x, low1.y}, {total_progress, -radius, segment_heading}});
                vertices.push_back({{low2.x, low2.y}, {total_progress + new_progress, -radius, segment_heading}});
                vertices.push_back({{high1.x, high1.y}, {total_progress, radius, segment_heading}});
                vertices.push_back({{high2.x, high2.y}, {total_progress + new_progress, radius, segment_heading}});

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

            // allow car to be before first frame
            {
                const GLuint ai = 2;
                const GLuint bi = 0;
                const GLuint ci = 4;

                const glm::fvec2 a = {vertices[ai].world.x, vertices[ai].world.y};
                const glm::fvec2 b = {vertices[bi].world.x, vertices[bi].world.y};
                const glm::fvec2 c = {vertices[ci].world.x, vertices[ci].world.y};

                const glm::fvec2 ac_unit = glm::normalize(c - a);
                const glm::fvec2 ac_norm = glm::fvec2(ac_unit.y, -ac_unit.x);

                if (glm::dot(car_pos - b, ac_norm) < 0) { // car is behind first triangles
                    const glm::fvec2 bcar = car_pos - b;
                    const glm::fvec2 car_parallel_plane = glm::normalize(glm::fvec2(bcar.y, -bcar.x));
                    const glm::fvec2 new_edge_center = b + bcar * (glm::length(bcar) + car_padding) / glm::length(bcar);

                    const glm::fvec2 v1_world = new_edge_center - car_parallel_plane * radius;
                    const glm::fvec2 v2_world = new_edge_center + car_parallel_plane * radius;

                    const float v1_progress = glm::dot( v1_world - b, ac_norm);
                    const float v2_progress = glm::dot(v2_world - b, ac_norm);

                    const float v1_offset = glm::dot(v1_world - b, ac_unit);
                    const float v2_offset = glm::dot(v2_world - b, ac_unit);

                    const float v1_heading = vertices[bi].curv.heading;
                    const float v2_heading = vertices[bi].curv.heading;

                    const Vertex v1 = {{v1_world.x, v1_world.y}, {v1_progress, v1_offset, v1_heading}};
                    const Vertex v2 = {{v2_world.x, v2_world.y}, {v2_progress, v2_offset, v2_heading}};

                    vertices.push_back(v1);
                    vertices.push_back(v2);

                    const GLuint v1i = vertices.size() - 2;
                    const GLuint v2i = vertices.size() - 1;

                    indices.push_back(v1i);
                    indices.push_back(ai);
                    indices.push_back(ci);

                    indices.push_back(v1i);
                    indices.push_back(ci);
                    indices.push_back(v2i);
                }
            }


            glBindBuffer(GL_ARRAY_BUFFER, m_gl_path.vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gl_path.ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_DYNAMIC_DRAW);
        }

        void StateEstimator_Impl::estimate_whl_speeds(float swangle) {
            const float xdot = m_world_state[state_car_xdot_idx];
            const float yawdot = m_world_state[state_yawdot_idx];

            const float whl_speed_f = (xdot * std::cos(swangle) + cg_to_front * yawdot * std::sin(swangle)) / whl_radius;
            const float whl_speed_r = xdot / whl_radius;

            m_world_state[state_whl_speed_f_idx] = whl_speed_f;
            m_world_state[state_whl_speed_r_idx] = whl_speed_r;
            std::cout << "whl_speed_f: " << whl_speed_f << " whl_speed_r: " << whl_speed_r << std::endl;
        }
    }
}
