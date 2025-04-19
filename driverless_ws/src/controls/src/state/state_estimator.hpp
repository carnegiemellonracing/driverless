#pragma once

#include <types.hpp>
#include <glm/glm.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <chrono>
#include <vector>


namespace controls {
    namespace state {
        /**
         * @brief State Estimator! Provides functions for controller node to use to prepare state information for mppi.
         * Implemnentation is in @ref StateEstimator_Impl.
         * Refer to the @rst :doc:`explainer </source/explainers/state_estimation>` @endrst for a more detailed overview.
         */
        class StateEstimator {

        public:
         /**
          * @brief Essentially serves as a named constructor that grants ownership of a StateEstimator to the caller.
          * This is necessary because StateEstimator is an abstract base class (a pointer is needed)
          * @param[in] mutex Reference to the mutex that state estimator will use. Prevents any public function from being called concurrently, which would cause undefined behavior of enabling the same CUDA context.
          * @param[in] logger Function object of logger (string to void)
          * @return Pointer to the created StateEstimator
          */

         static std::shared_ptr<StateEstimator> create(std::mutex& mutex, LoggerFunc logger = no_log);
            bool m_follow_midline_only = false;
            void set_follow_midline_only(bool follow_midline_only) {
                m_follow_midline_only = follow_midline_only;
            }
            
            virtual State project_state(const rclcpp::Time &time) =0;

            /**
             * @brief "main" function of the state estimator. Calculates current inertial state and the
             * inertial-to-curvilinear lookup table, then syncs them to the GPU for MPPI use.
             * (into @ref cuda_globals::curv_frame_lookup_tex and @ref cuda_globals::curr_state respectively)
             * @param time The current time
             */
            virtual std::vector<std::chrono::milliseconds> sync_to_device(const rclcpp::Time &time) =0;

            virtual void render_and_sync(State state) =0;

            /**
             * @brief Callback for spline messages. Updates the state estimator with the new spline. Used for both
             * state projection and curvilinear state lookup table generation.
             * @param spline_msg The spline message
             */
            virtual void on_spline(const SplineMsg& spline_msg) =0;

            virtual float on_cone(const ConeMsg& cone_msg) =0;

         /**
          * @brief Callback for twist messages. Updates the state estimator with the new twist. Used for state projection.
          * @param twist_msg The spline message
          * @param time The time the twist is accurate to.
          */
         virtual void on_twist(const TwistMsg& twist_msg, const rclcpp::Time &time) =0;

            /**
            * @brief Callback for slam_pose messages. Updates the state estimator with the new pose from slam
            * @param slam_pose The slam_pose message
            */
            virtual void on_slam_pose(const SlamPoseMsg& slam_pose) =0;

        
            /**      
            * @brief Callback for slam messages. Updates the state estimator with the new slam message.
            * @param slam_msg The slam message
            * @param time The time the slam message is accurate to.
            */


          virtual void on_slam(const SlamMsg& slam_msg,rclcpp::Time time) =0;
        



         /**
          * @brief Callback for pose messages. Can be used for state projection, but currently unused.
          * Reason: position given by spline instead, yaw given by steering wheel angle. Pose meanwhile is noisy?
          * @param pose_msg The pose message
          */
         virtual void on_pose(const PoseMsg& pose_msg) =0;

            /**
             * @brief Returns whether state projection is ready. If not, the MPPI controller should wait. This should
             * only be not ready when the state estimator is first initialized and the controller has no spline.
             * @return True if state projection is ready, False otherwise.
             */
            virtual bool is_ready() =0;

            /**
             * @brief Sets the logger function for the state estimator. Can be integrated with ROS or print to stdout.
             * @param logger The logger function to be used.
             */
            virtual void set_logger(LoggerFunc logger) =0;

         /**
          * Attaches the node's logger object to the state estimator.
          * @param logger The logger object to be bound.
          */
         virtual void set_logger_obj(rclcpp::Logger logger) =0;
            /**
             * @brief Records a control action for state projection purposes.
             * @param action The control action to record
             * @param ros_time The time the action was recorded
             */
            virtual void record_control_action(const Action &action, const rclcpp::Time &ros_time) =0;

            virtual std::vector<glm::fvec2> get_spline_frames() = 0;

#ifdef DISPLAY
            struct OffsetImage {
                std::vector<float> pixels;
                uint pix_width;
                uint pix_height;
                glm::fvec2 center;
                float world_width;
            };

            virtual std::vector<glm::fvec2> get_all_left_cone_points() =0;
            virtual std::vector<glm::fvec2> get_all_right_cone_points() =0;
            virtual std::unordered_map<uint32_t, std::pair<std::vector<glm::fvec2>, std::vector<glm::fvec2>>> get_slam_chunks() = 0;
            virtual std::vector<glm::fvec2> get_left_cone_points() = 0;
            virtual std::vector<glm::fvec2> get_right_cone_points() = 0;
            virtual std::vector<glm::fvec2> get_raceline_points() =0;

            virtual std::pair<std::vector<glm::fvec2>, std::vector<glm::fvec2>> get_all_cone_points() =0;
            virtual std::vector<float> get_vertices() =0;
            // virtual std::vector<GLuint> get_indices()=0;
            virtual void get_offset_pixels(OffsetImage& offset_image) =0;
#endif

            /**
             * @brief Virtual destructor for the state estimator.
             */
            virtual ~StateEstimator() = default;
        };
    }
}
