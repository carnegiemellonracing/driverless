#pragma once

#include <cuda_globals/cuda_globals.cuh>
#include <thrust/device_vector.h>
#include <utils/gl_utils.hpp>

#include "state_estimator.hpp"


namespace controls {
    namespace state {

        class StateProjector {
        public:
            /**
             * @brief Record an action into the history
             * @param[in] action The action to be recorded
             * @param[in] time The time at which the action was taken by the actuators.
             */
            void record_action(Action action, rclcpp::Time time);

            /**
             * @brief Record a speed into the history
             * @param[in] speed The speed to be recorded
             * @param[in] time The time at which the speed was measured. Should have no latency.
             */
            void record_speed(float speed, rclcpp::Time time);

            /**
             * @brief Record a pose into the history. Although pose data itself is not measured, this is called when
             * the spline is updated, and a pose of (0,0,0) is inferred from the spline.
             * @param x The x coordinate of the pose
             * @param y The y coordinate of the pose
             * @param yaw The yaw of the vehicle w.r.t. the inertial coordinate frame
             * @param time The time at which the vehicle had the pose. Since the pose is inferred from the spline, this
             * should be when the LIDAR points first came in.
             */
            void record_pose(float x, float y, float yaw, rclcpp::Time time,int32_t slam_id);

            /**
             * @brief "main" projection function. Projects the state of the car at a given time, from the most
             * recent pose record and the history of actions and speeds since that pose record.
             * @param time The time at which the state is to be projected
             * @param logger The logger function to be used
             * @return The projected state of the car at the given time
             */
            State project(const rclcpp::Time& time, LoggerFunc logger) const;
            /**
             * @brief Whether the StateProjector is ready to project a state. This is true if there is a pose record,
             * which is every time since the first spline is received.
             * @return True if the StateProjector is ready to project a state, false otherwise.
             */
            bool is_ready() const;

        private:
            /// Historical record type
            struct Record {
            enum class Type {
                Action,
                Speed,
                SlamPose
            };

            union {
                Action action;
                float speed;

                struct {
                float x;
                float y;
                float yaw;
                } pose;

                struct {
                int32_t current_chunk_id;
                struct {
                    float x;
                    float y;
                    float yaw;
                } pose;
                } slamPose;
            };

            rclcpp::Time time;
            Type type;
            };

            /// Prints the elements of m_history_since_pose, for debugging purposes.
            void print_history() const;

            /// @note m_init_action and m_init_speed should occur <= m_pose_record, if m_pose_record exists

            /// "Default" action to be used until a recorded action is available
            Record m_init_action { .action {}, .time = rclcpp::Time(0UL, default_clock_type), .type = Record::Type::Action};
            /// "Default" speed to be used until a recorded speed is available
            /// @note direction of velocity is inferred from swangle
            Record m_init_speed { .speed = 0, .time = rclcpp::Time(0UL, default_clock_type), .type = Record::Type::Speed};
            /// most recent and only pose (new pose implies a new coord. frame, throw away data in old coord. frame)
            /// only nullopt before first pose received
            std::optional<Record> m_pose_record = std::nullopt;
            std::optional<Record> m_slam_pose_record = std::nullopt;
            
            /// Helper binary operator for sorting records by time, needed for the multiset.
            struct CompareRecordTimes {
                bool operator() (const Record& a, const Record& b) const {
                    return a.time < b.time;
                }
            };
            /**
             * Contains all action and speed records since the last pose record.
             * Multiset is used like a self-sorting array.
             *
             * Invariants of m_history_since_pose:
             * should only contain Action and Speed records
             * time stamps of each record should be strictly after m_pose_record
             */
            std::multiset<Record, CompareRecordTimes> m_history_since_pose {};
        };

        class StateEstimator_Impl : public StateEstimator {
        public:
            /**
             * @brief Actual constructor, called implicitly by make_shared
             * @param mutex StateEstimator's mutex
             * @param logger logger function to be used
             */
            StateEstimator_Impl(std::mutex& mutex, LoggerFunc logger);

            void on_spline(const SplineMsg& spline_msg) override;
            float on_cone(const ConeMsg& cone_msg) override;
            float on_slam_pose(const SlamPoseMsg& slam_pose_msg) override;
            void on_slam(const SlamMsg& slam_msg, const rclcpp::Time &time) override;
            void on_twist(const TwistMsg& twist_msg, const rclcpp::Time &time) override;
            // on_pose is not used, for future proofing
            void on_pose(const PoseMsg& pose_msg) override;

            void render_and_sync(State state) override;
            State project_state(const rclcpp::Time &time) override;
            std::vector<std::chrono::milliseconds> sync_to_device(const rclcpp::Time &time) override;
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
            std::unordered_map<int32_t, std::pair<std::vector<glm::fvec2>, std::vector<glm::fvec2>>> get_slam_chunks() const override {
                std::lock_guard<std::mutex> guard {m_mutex};
                return m_slam_chunks;
            }
            std::vector<glm::fvec2> get_raceline_points();
            std::pair<std::vector<glm::fvec2>, std::vector<glm::fvec2>> get_all_cone_points() override;
            std::vector<float> get_vertices() override;
            // std::vector<glm::fvec2> get_normals() override;
            void get_offset_pixels(OffsetImage& offset_image) override;
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

            std::mutex& m_mutex;  // Reference to the mutex
            std::unordered_map<int32_t, std::pair<std::vector<glm::fvec2>, std::vector<glm::fvec2>>> m_slam_chunks;

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
            StateProjector m_state_projector;

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
        };

    }
}
