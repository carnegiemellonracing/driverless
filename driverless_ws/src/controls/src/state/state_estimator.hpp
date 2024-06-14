#pragma once

#include <types.hpp>
#include <glm/glm.hpp>
#include <builtin_interfaces/msg/time.hpp>


namespace controls {
    namespace state {
        /**
         * @brief State Estimator!
         * In a nutshell, given twist and spline information through ROS messages, it estimates:
         * - the current inertial state of the vehicle (by using our model to project from stale data)
         * - a lookup table from inertial state to curviliniear state based on the spline and OpenGL magic
         * Then it syncs these to the GPU for the MPPI controller to use.
         */
        class StateEstimator {
        public:
         /**
          * @brief Essentially serves as a named constructor that grants ownership of a StateEstimator to the caller.
          * This is necessary because StateEstimator is an abstract base class (a pointer is needed)
          * @param[in] mutex Reference to the mutex that state estimator will use TODO: who else uses it? prevents any public function from being called concurrently - enabling same CUDA context (undefined)
          * @param[in] logger Function object of logger (string to void) TODO: is function object correct?
          * @return Pointer to the created StateEstimator
          */
         static std::shared_ptr<StateEstimator> create(std::mutex& mutex, LoggerFunc logger = no_log);

            /**
             * @brief "main" function of the state estimator. Calculates current inertial state and the
             * inertial-to-curvilinear lookup table, then syncs them to the GPU for MPPI use.
             * (into @ref cuda_globals::curv_frame_lookup_tex and @ref cuda_globals::curr_state respectively)
             * @param time The current time
             */
            virtual void sync_to_device(const rclcpp::Time &time) =0;

            /**
             * @brief Callback for spline messages. Updates the state estimator with the new spline. Used for both
             * state projection and curvilinear state lookup table generation.
             * @param spline_msg The spline message
             */
            virtual void on_spline(const SplineMsg& spline_msg) =0;
         /**
          * @brief Callback for twist messages. Updates the state estimator with the new twist. Used for state projection.
          * @param twist_msg The spline message
          * @param time @TODO why does this need to be passed in?
          */
         virtual void on_twist(const TwistMsg& twist_msg, const rclcpp::Time &time) =0;

         /**
          * @brief Callback for pose messages. Can be used for state projection, but currently unused.
          * Reason: position given by spline instead, yaw given by steering wheel angle. Pose meanwhile is noisy?
          * @param pose_msg The pose message
          */
         virtual void on_pose(const PoseMsg& pose_msg) =0;

            /**
             * @brief Returns whether state projection is ready. If not, the MPPI controller should wait. This should
             * only be not ready when the state estimator is first initialized.
             * @return True if state projection is ready, False otherwise.
             */
            virtual bool is_ready() =0;
            /**
             * @brief Esimates and returns current state by projecting from last known twist/spline information.
             * @return The current estimated state
             * @note Spline provides state information since car is at (0, 0) relative to a new spline.
             * @TODO Why must this be a public function? For debugging reasons?
             */
            virtual State get_projected_state() =0;
            /**
             * @brief Sets the logger function for the state estimator. Can be integrated with ROS or print to stdout.
             * @param logger The logger function to be used.
             */
            virtual void set_logger(LoggerFunc logger) =0;
            /**
             * @brief Returns the timestamp of the last spline message received.
             * @return The timestamp of the last spline message received.
             * @TODO verify copilot isnt trolling me
             */
            virtual rclcpp::Time get_orig_spline_data_stamp() =0;
            /**
             * @brief Records a control action for state projection purposes.
             * @param action The control action to record
             * @param ros_time The time the action was recorded
             */
            virtual void record_control_action(const Action &action, const rclcpp::Time &ros_time) =0;

#ifdef DISPLAY
            struct OffsetImage {
                std::vector<float> pixels;
                uint pix_width;
                uint pix_height;
                glm::fvec2 center;
                float world_width;
            };

            virtual std::vector<glm::fvec2> get_spline_frames() =0;
            virtual void get_offset_pixels(OffsetImage& offset_image) =0;
#endif

            /**
             * @brief Virtual destructor for the state estimator.
             */
            virtual ~StateEstimator() = default;
        };
    }
}
