#pragma once

#include <cuda_globals/cuda_globals.cuh>
#include <thrust/device_vector.h>
#include <utils/gl_utils.hpp>

#include "state_estimator.hpp"
#include "state_projector.cuh"

namespace controls {
    namespace state {

        class StateEstimator_Impl : public StateEstimator {
        public:
            /**
             * @brief Actual constructor, called implicitly by make_shared
             * @param mutex StateEstimator's mutex
             * @param logger logger function to be used
             */
            StateEstimator_Impl(std::mutex& mutex, LoggerFunc logger);

            void on_spline(const SplineMsg& spline_msg) override;
            void on_quat(const QuatMsg& quat_msg) override;
            float on_cone(const ConeMsg& cone_msg) override;
            void on_twist(const TwistMsg& twist_msg, const rclcpp::Time &time) override;
            // on_pose is not used, for future proofing
            void on_pose(const PoseMsg& pose_msg) override;
            void on_position_lla(const PositionLLAMsg& position_lla_msg) override;


            void render_and_sync(State state) override;
            std::optional<State> project_state(const rclcpp::Time &time) override;
            bool is_ready() override;
            void set_logger(LoggerFunc logger) override;
            void set_logger_obj(rclcpp::Logger logger) override;

            void record_control_action(const Action &action, const rclcpp::Time &time) override;

            std::vector<glm::fvec2> get_spline_frames() override;


#ifdef DISPLAY
            std::vector<glm::fvec2> get_all_left_cone_points() override;
            std::vector<glm::fvec2> get_all_right_cone_points() override;
            std::vector<glm::fvec2> get_left_cone_points() override;
            std::vector<glm::fvec2> get_right_cone_points() override;
            std::vector<glm::fvec2> get_raceline_points();
            std::vector<float> get_vertices() override;
            // std::vector<glm::fvec2> get_normals() override;
            OffsetImage get_offset_pixels() override;
            OffsetImage m_last_offset_image;

#endif

            /**
             * @brief Destructor, cleans up OpenGL resources
             */
            ~StateEstimator_Impl() override;

        private:
            /// Index into where the uniform scale is stored for vertex shader to reference.
            constexpr static GLint shader_scale_loc = 0;
            /// Index into where the uniform center is stored for vertex shader to reference.
            constexpr static GLint shader_center_loc = 1;

            /// Calculates scale and center transformation from world to lookup table, stores
            /// in m_curv_frame_lookup_tex_info.
            void gen_tex_info(glm::fvec2 car_pos);

            void render_fake_track();
            /**
             * Render the lookup table into m_curv_frame_lookup_fbo
             */
            void render_curv_frame_lookup();
            /**
             * Maps the rendered OpenGL frame buffer to a CUDA texture (cuda_globals::curv_frame_lookup_tex)
             * Since the OpenGL frame buffer can't be referred to by the rest of the code directly.
             */
            void map_curv_frame_lookup();
            /**
             * Unmaps the frame buffer from CUDA memory, since
             * you can't render to something when it is mapped.
             * Think about it like releasing a mutex.
             */
            void unmap_curv_frame_lookup();

            /**
             * @brief Syncs m_curv_frame_lookup_tex_info to cuda_globals so it can be accessed by MPPI
             */
            void sync_tex_info();
            /**
             * @brief Creates the frame and render buffers (m_curv_frame_lookup_fbo/rbo).
             */
            void gen_curv_frame_lookup_framebuffer();

            void gen_fake_track();

            /**
             * Creates the buffers to be used, as well as the descriptions of how the buffers are laid out.
             * @brief Creates the names for the vao, vbo and ebo.
             * Specifies how the vbo should be laid out, stores this in the vao.
             * Lastly, binds to the ebo.
             */
            void gen_gl_path(utils::GLObj &gl_path);

            /**
             * @brief Fills in the vertex and element buffers with the actual vertex position information
             * and the triples of vertex indices that represent triangles respectively.
             */
            void fill_path_buffers_cones();
            void fill_path_buffers_spline();

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
        

            /// Stores the sequence of (x,y) spline points from path planning.
            std::vector<glm::fvec2> m_spline_frames;

            std::vector<glm::fvec2> m_all_left_cone_points;
            std::vector<glm::fvec2> m_all_right_cone_points;
            std::vector<glm::fvec2> m_raceline_points;

            std::vector<glm::fvec2> m_left_cone_points;
            std::vector<glm::fvec2> m_right_cone_points;

            std::vector<float> m_vertices;
            std::vector<GLuint> m_triangles;

            int m_num_triangles;

            /// The owned StateProjector to estimate the current state.
            state::StateProjector m_state_projector;

            ///// -----OPENGL OBJECTS----- /////

            cudaGraphicsResource_t m_curv_frame_lookup_rsc;
            /// Center and scale information (transformation from IRL to rendered coordinates)
            cuda_globals::CurvFrameLookupTexInfo m_curv_frame_lookup_tex_info;            
            /// Frame buffer object. Where the OpenGL shader pipeline renders to. The actual lookup table.
            /// OpenGL object, not a CUDA object.
            GLuint m_curv_frame_lookup_fbo;
            /**
             * TODO:
             * 1. Make the fbo complete
             * 2. Check for completeness
             * 3. Delete the framebuffer after use
             */
            /// Render buffer object. Render target for the fbo, used to map to CUDA memory
            GLuint m_curv_frame_lookup_rbo;

            GLuint m_fake_track_fbo;
            GLuint m_fake_track_texture_color;
            utils::GLObj m_fake_track_path;
            GLuint m_fake_track_shader_program; 

            /**
             * Has members vbo, vao, veo. @ref utils::GLObj
             * vbo - vertex buffer object. Contiguous block of memory where all the vertices' information is stored
             * vao - vertex array object. A specification for how memory in the VBO is laid out, e.g. how big is a vertex.
             * ebo - element buffer object. Memory that contains triples of indices into the VBO, representing triangles.
             */
            utils::GLObj m_gl_path;

            GLuint m_gl_path_shader; ///< Shader program to be used. Composed of vertex and fragment shaders.

            /// OpenGL context, stores all the information associated with this instance.
            SDL_GLContext m_gl_context;
            /// OpenGL window assoicated with the context
            SDL_Window* m_gl_window;

            /// Whether the frame buffer is mapped to CUDA memory or not
            bool m_curv_frame_lookup_mapped = false;

            LoggerFunc m_logger; ///< The logger function to be used.
            rclcpp::Logger m_logger_obj; ///< Logger object (belonging to the node)
            std::mutex& m_mutex; ///< Mutex to prevent multiple method calls from happening simultaneously
        };

    }
}
