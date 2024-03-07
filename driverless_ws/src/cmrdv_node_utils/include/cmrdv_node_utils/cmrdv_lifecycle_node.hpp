#ifndef CMRDV_NODE_UTILS__CMRDV_LIFECYCLE_NODE_HPP_
#define CMRDV_NODE_UTILS__CMRDV_LIFECYCLE_NODE_HPP_

#include <memory>
#include <string>
#include <thread>
#include <boost/optional.hpp>

#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"


namespace cmrdv_node_utils 
{

    using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

    /**
     *  Provides default exception handling and system alert handling for components in cmrdv.
     *  All new nodes are expected to be built off CMRDVLifecycleNode and implemented as components.
     * 
     * TODO: UPDATE LATER
     * Every function which uses callbacks (subscribers, publishers, services, timers) is wrapped in a try catch block
     *      In the event an exception makes it to this class it will result in a SystemAlert message being published and the node shutting down.
     * 
     * Each lifecycle node has a heartbeat publisher. In the event this node is not able to publish a 'safe' heartbeat status 
     *      it will result in a SystemAlert message being published and the node shutting down.
     * 
     * The user can add their own exception, alert, and shutdown handling using corresponding handler functions:
     *      handle_on_error, handle_on_system_alert, handle_on_shutdown
     * 
    */
    class CMRDVLifecycleNode : public rclcpp_lifecycle::LifecycleNode 
    {

        public:

            /**
             * \brief Constructor which is used by component loaders
             * 
             * \param options The configurations options for this node
             */
            // CMRDV_NODE_UTILS_PUBLIC --->EXPLORE VISIBILITY
            explicit CMRDVLifecycleNode(const rclcpp::NodeOptions &options);


            /**
             * \brief Destructor. Extending classes should override this
             */
            virtual ~CMRDVLifecycleNode();


            // 

            /**
             * \brief Software Architecture Implementation (DO NOT OVERRIDE)
             * 
             * \note use corresponding "handle_<method>" function instead.
             * \note Overrides: See https://github.com/ros2/rclcpp/blob/foxy/rclcpp_lifecycle/include/rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp for their details
             */
            cmrdv_node_utils::CallbackReturn on_configure(const rclcpp_lifecycle::State &prev_state) override;
            /**
             * \brief Software Architecture Implementation (DO NOT OVERRIDE)
             * 
             * \note use corresponding "handle_<method>" function instead.
             * \note Overrides: See https://github.com/ros2/rclcpp/blob/foxy/rclcpp_lifecycle/include/rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp for their details
             */
            cmrdv_node_utils::CallbackReturn on_activate(const rclcpp_lifecycle::State &prev_state) override;
            /**
             * \brief Software Architecture Implementation (DO NOT OVERRIDE)
             * 
             * \note use corresponding "handle_<method>" function instead.
             * \note Overrides: See https://github.com/ros2/rclcpp/blob/foxy/rclcpp_lifecycle/include/rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp for their details
             */
            cmrdv_node_utils::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &prev_state) override;
            /**
             * \brief Software Architecture Implementation (DO NOT OVERRIDE)
             * 
             * \note use corresponding "handle_<method>" function instead.
             * \note Overrides: See https://github.com/ros2/rclcpp/blob/foxy/rclcpp_lifecycle/include/rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp for their details
             */
            cmrdv_node_utils::CallbackReturn on_cleanup(const rclcpp_lifecycle::State &prev_state) override;
            /**
             * \brief Software Architecture Implementation (DO NOT OVERRIDE)
             * 
             * \note use corresponding "handle_<method>" function instead.
             * \note Overrides: See https://github.com/ros2/rclcpp/blob/foxy/rclcpp_lifecycle/include/rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp for their details
             */
            cmrdv_node_utils::CallbackReturn on_error(const rclcpp_lifecycle::State &prev_state) override;
            /**
             * \brief Software Architecture Implementation (DO NOT OVERRIDE)
             * 
             * \note use corresponding "handle_<method>" function instead.
             * \note Overrides: See https://github.com/ros2/rclcpp/blob/foxy/rclcpp_lifecycle/include/rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp for their details
             */
            cmrdv_node_utils::CallbackReturn on_shutdown(const rclcpp_lifecycle::State &prev_state) override;


            /**
             * \brief Callback triggered when transitioning from UNCONFIGURED to INACTIVE.
             *        This method should be overriden to load parameters and setup callbacks
             * 
             * \param state The previous state 
             * 
             * \return A callback success flag. If SUCCESS then the state will transition to INACTIVE. 
             *        If FAILURE or ERROR occurs the state will transition to ErrorProcessing via the handle_on_error method
             */
            virtual cmrdv_node_utils::CallbackReturn handle_on_configure(const rclcpp_lifecycle::State &prev_state);


            /**
             * \brief Callback triggered when transitioning from INACTIVE to ACTIVE
             *        This method should be overriden to set any states or callbacks that should only trigger in the ACTIVE state
             * 
             * \param state The previous state 
             * 
             * \return A callback success flag. If SUCCESS then the state will transition to ACTIVE. 
             *        If FAILURE or ERROR occurs the state will transition to ErrorProcessing via the handle_on_error method
             */
            virtual cmrdv_node_utils::CallbackReturn handle_on_activate(const rclcpp_lifecycle::State &prev_state);
    

            /**
             * \brief Callback triggered when transitioning from ACTIVE to INACTIVE
             *        This method should be overriden to clear any runtime state data that would prevent reconfiguring of the node
             *        and or deactivate any timers/callbacks that should only trigger in the ACTIVE state
             * 
             * \note CMRDVLifecycleNode will automatically deactivate publishers on this callback. 
             * 
             * \param state The previous state 
             * 
             * \return A callback success flag. If SUCCESS then the state will transition to INACTIVE. 
             *        If FAILURE or ERROR occurs the state will transition to ErrorProcessing via the handle_on_error method
             */
            virtual cmrdv_node_utils::CallbackReturn handle_on_deactivate(const rclcpp_lifecycle::State &prev_state);
        

            /**
             * \brief Callback triggered when transitioning from INACTIVE to UNCONFIGURED
             *        This method should be overriden to clear any state and memory allocation that occurred.
             *        When completed the node should be fully unconfigured.
             * 
             * \note CMRDVLifecycleNode will automatically deactivate publishers and clear timer pointers
             * 
             * \param state The previous state 
             * 
             * \return A callback success flag. If SUCCESS then the state will transition to UNCONFIGURED. 
             *        If FAILURE or ERROR occurs the state will transition to ErrorProcessing via the handle_on_error method
             */
            virtual cmrdv_node_utils::CallbackReturn handle_on_cleanup(const rclcpp_lifecycle::State &prev_state);


            /**
             * \brief Callback triggered when transitioning from any state to ShuttingDown.
             *        This method should be overriden to add any shutdown logic
             * 
             * 
             * \param state The previous state 
             * 
             * \return A callback success flag. If SUCCESS then the state will transition to FINALIZED. 
             *        If FAILURE or ERROR occurs the state will transition to FINALIZED and the process will shutdown
             */
            virtual cmrdv_node_utils::CallbackReturn handle_on_shutdown(const rclcpp_lifecycle::State &prev_state);


            /**
             * \brief Callback triggered when transitioning from any state to ErrorProcessing due to the error signal.
             *        This method should be overriden to add any exception handling logic.
             * 
             *  \note CMRDVLifecycleNode will automatically publish a SystemAlert before this is triggered. 
             * 
             * \param state The previous state 
             * 
             * \return A callback success flag. If SUCCESS then the state will transition to UNCONFIGURED. 
             *        If FAILURE or ERROR occurs the state will transition to FINALIZED and the process will shutdown
             */
            virtual cmrdv_node_utils::CallbackReturn handle_on_error(const rclcpp_lifecycle::State &prev_state, const std::string &exception_string);


            /**
             * \brief Convenience method to build a shared pointer from this object.
             * 
             * \return A shared pointer which points to this object
             */
            std::shared_ptr<cmrdv_node_utils::CMRDVLifecycleNode> shared_from_this();


            // TODO: create more of these type of functions
            /**
             * \brief Publishes a SystemAlert message to the rest of the cmrdv node system
             * \note This callback will automatically populate the msg.source_node field based on this node name.
             * \param msg The message to publish
             */
            // TODO: setup cmrdv_msgs
            // void publish_system_alert(const cmrdv_msgs::msg::SystemAlert &msg);


            /**
             * \brief Override of parent method. See descriptive comments here:
             *  https://github.com/ros2/rclcpp/blob/4859c4e43576d0c6fe626679b2c2604a9a8b336c/rclcpp_lifecycle/include/rclcpp_lifecycle/lifecycle_node.hpp#L230
             * 
             * \note The function object passed to this method will be moved using std::move. 
             *       The user should therefore assume ownership of this function object has been relinquished
             */
            template <
                typename MessageT,
                typename CallbackT,
                typename AllocatorT = std::allocator<void>,
                typename CallbackMessageT =
                    typename rclcpp::subscription_traits::has_message_type<CallbackT>::type,
                typename SubscriptionT = rclcpp::Subscription<MessageT, AllocatorT>,
                typename MessageMemoryStrategyT = rclcpp::message_memory_strategy::MessageMemoryStrategy<
                    CallbackMessageT,
                    AllocatorT>>
            std::shared_ptr<SubscriptionT>
            create_subscription(
                const std::string &topic_name,
                const rclcpp::QoS &qos,
                CallbackT &&callback, // temporary variable (can be lambda or std::bind)
                const rclcpp_lifecycle::SubscriptionOptionsWithAllocator<AllocatorT> &options =
                rclcpp_lifecycle::create_default_subscription_options<AllocatorT>(),
                typename MessageMemoryStrategyT::SharedPtr msg_mem_strat = (MessageMemoryStrategyT::create_default()));


            /**
             * \brief Override of parent method. See descriptive comments here:
             *  https://github.com/ros2/rclcpp/blob/4859c4e43576d0c6fe626679b2c2604a9a8b336c/rclcpp_lifecycle/include/rclcpp_lifecycle/lifecycle_node.hpp#L201
             */
            template <typename MessageT, typename AllocatorT = std::allocator<void>>
            std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<MessageT, AllocatorT>>
            create_publisher(
                const std::string &topic_name,
                const rclcpp::QoS &qos,
                const rclcpp_lifecycle::PublisherOptionsWithAllocator<AllocatorT> &options = (rclcpp_lifecycle::create_default_publisher_options<AllocatorT>()));


            /**
             * \brief Override of parent method. See descriptive comments here:
             *  https://github.com/ros2/rclcpp/blob/4859c4e43576d0c6fe626679b2c2604a9a8b336c/rclcpp_lifecycle/include/rclcpp_lifecycle/lifecycle_node.hpp#L463
             * 
             * \note The function object passed to this method will be moved using std::move. 
             *       The user should therefore assume ownership of this function object has been relinquished
             */
            rclcpp_lifecycle::LifecycleNode::OnSetParametersCallbackHandle::SharedPtr
            add_on_set_parameters_callback(
                rclcpp_lifecycle::LifecycleNode::OnParametersSetCallbackType callback);


            /**
             * \brief Override of parent method. See descriptive comments here:
             *  https://github.com/ros2/rclcpp/blob/4859c4e43576d0c6fe626679b2c2604a9a8b336c/rclcpp_lifecycle/include/rclcpp_lifecycle/lifecycle_node.hpp#L249
             *
             * \note The function object passed to this method will be moved using std::move. 
             *       The user should therefore assume ownership of this function object has been relinquished
             */
            template <typename DurationRepT = int64_t, typename DurationT = std::milli, typename CallbackT>
            std::shared_ptr<rclcpp::TimerBase> // NOTE: return value of TimerBase must be used here to account for the fact that the exception handling lambda will have a different type from the input callback type due to being a possible differnet location in code (member vs non-member method etc.).
                                            // Therefore the old return statement of typename rclcpp::WallTimer<CallbackT>::SharedPtr is replaced with TimerBase
            create_wall_timer(
                std::chrono::duration<DurationRepT, DurationT> period,
                CallbackT callback,
                rclcpp::CallbackGroup::SharedPtr group = nullptr);


            /**
             * \brief Method to create a timer whose lifecycle can be managed by this node.
             *  
             *  \note: In foxy the LifecycleNode api is slightly out of sync with the node api so there is not a create_timer method there. We use rclcpp directly here
             *  
             *  \param clock The underlying clock to use for the timer.
             *  \param period The period of trigger of the timer.
             *  \param callback The callback to execute on timer trigger
             *  \param group The callback group to use for processing the callback.
             * 
             *  \return A pointer to an intialized timer. The timer will be cancled when this node transitions through a deactivate/cleanup sequence
             *
             * \note The function object passed to this method will be moved using std::move. 
             *       The user should therefore assume ownership of this function object has been relinquished
             * 
             */
            template <typename CallbackT>
            typename rclcpp::TimerBase::SharedPtr
            create_timer(
                rclcpp::Clock::SharedPtr clock,
                rclcpp::Duration period,
                CallbackT &&callback,
                rclcpp::CallbackGroup::SharedPtr group = nullptr);


            /**
             * \brief Override of rclcpp method. See descriptive comments here:
             *  https://github.com/ros2/rclcpp/blob/4859c4e43576d0c6fe626679b2c2604a9a8b336c/rclcpp_lifecycle/include/rclcpp_lifecycle/lifecycle_node.hpp#L271
             *  
             * \note In foxy the LifecycleNode api is slightly out of sync with the node api so there is not a create_timer method there. We use rclcpp directly here
             *
             * \note The function object passed to this method will be moved using std::move. 
             *       The user should therefore assume ownership of this function object has been relinquished
             */
            template <typename ServiceT, typename CallbackT>
            typename rclcpp::Service<ServiceT>::SharedPtr
            create_service(
                const std::string &service_name,
                CallbackT &&callback,
                const rmw_qos_profile_t &qos_profile = rmw_qos_profile_services_default,
                rclcpp::CallbackGroup::SharedPtr group = nullptr);


            /**
             * \brief Override of parent method. See descriptive comments here:
             *  https://github.com/ros2/rclcpp/blob/d7804e1b3fd9676d302ec72f02c49ba04cbed5e6/rclcpp_lifecycle/include/rclcpp_lifecycle/lifecycle_node.hpp#L260
             * 
             * VERY IMPORTANT \note While the callback group default appears to be nullptr this is not actually the case. 
             *                      On call the group will be set to the Reentrant CallbackGroup this->service_callback_group_ 
             *                      This group is created explicitly for services to support a synchronous callback paradigm similar to ROS1. 
             *                      Care should be taken when changing this default group value.
             *                      If the user needs nullptr as the actual input they can call the parent method rclcpp_lifecycle::LifecycleNode::create_client
             */
            template <class ServiceT>
            typename rclcpp::Client<ServiceT>::SharedPtr
            create_client(const std::string service_name, 
                const rmw_qos_profile_t & qos_profile = rmw_qos_profile_services_default,
                rclcpp::CallbackGroup::SharedPtr group = nullptr);


        protected:

            // TODO: Helpful methods for all subteams

            /**
             * \brief Helper method to handle exceptions which occurred in primary states.
             *        This is needed as the life cycle diagram described here https://design.ros2.org/articles/node_lifecycle.html
             *        does not reflect current implementation where error handling is only handled by transition states. 
             *        Current ROS2 Issue/PR: https://github.com/ros2/rclcpp/pull/1064
             * 
             * \param e The exception to be handled
             */
            void handle_primary_state_exception(const std::exception &e);



            /**
             * \brief Helper method to publish a system alert with the provided error string.
             * \todo Setup alert levels
             * 
             * \param alert_string The message description to send.
             */
            void send_error_alert_msg_for_string(const std::string &alert_string);

            
            /**
             * \brief Activate all publishers to allow publication
             */
            void activate_publishers();


            /**
             * \brief Deactivate all publishers to prevent publication
             */
            void deactivate_publishers();


            /**
             * \brief Reset all publisher pointers
             */
            void cleanup_publishers();


            /**
             * \brief Stop and then reset all timer pointers
             */
            void cleanup_timers();


            // TODO: REFACTOR FOR CMRDV
            //! Topic to subscribe to by default for SystemAlert messages
            const std::string system_alert_topic_{"/system_alert"};


            //! System alert publisher
            // std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<cmrdv_msgs::msg::SystemAlert>>
                // system_alert_pub_;


            //! A list of lifecycle publishers produced from this node whose lifetimes can be managed
            std::vector<std::shared_ptr<rclcpp_lifecycle::LifecyclePublisherInterface>> lifecycle_publishers_;


            //! A list of timers produced from this node whose lifetimes can be managed
            std::vector<std::shared_ptr<rclcpp::TimerBase>> timers_;


            //! Mutex for use with exception handling
            std::mutex exception_mutex_;


            //! Optional caught exception description which serves as a workaround for adding exception handling to primary states
            boost::optional<std::string> caught_exception_;


            //! Reentrant callback group to use with service calls. Setup this way so that this class' functions
            //  can be called from topic callbacks
            rclcpp::CallbackGroup::SharedPtr service_callback_group_;


        private:
            // TODO: setup cmrdv error handlers and heartbeat system

    };

} // namespace cmrdv_node_utils     

// Template functions cannot be linked unless the implementation is provided
// Therefore include implementation to allow for template functions
#include "internal/cmrdv_lifecycle_node.tpp"

#endif // CMRDV_NODE_UTILS__CMRDV_LIFECYCLE_NODE_HPP_"