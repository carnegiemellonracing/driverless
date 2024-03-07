
#ifdef CMRDV_NODE_UTILS__CMRDV_LIFECYCLE_NODE_HPP_


namespace cmrdv_node_utils {

	template <typename MessageT, typename AllocatorT = std::allocator<void>>
	std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<MessageT, AllocatorT>>
	CMRDVLifecycleNode::create_publisher(
		const std::string &topic_name,
		const rclcpp::QoS &qos,
		const rclcpp_lifecycle::PublisherOptionsWithAllocator<AllocatorT> &options)
	{
		// default to ros2 implementation
		auto pub = rclcpp_lifecycle::LifecycleNode::create_publisher<MessageT>(topic_name, qos, options);
		// Change: add this publisher to list of publishers 
		lifecycle_publishers_.push_back(pub);
		return pub;
	}


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
	CMRDVLifecycleNode::create_subscription(
		const std::string &topic_name,
		const rclcpp::QoS &qos,
		CallbackT &&callback, // temporary variable (can be lambda or std::bind)
		const rclcpp_lifecycle::SubscriptionOptionsWithAllocator<AllocatorT> &options,
		typename MessageMemoryStrategyT::SharedPtr msg_mem_strat)
	{

		return rclcpp_lifecycle::LifecycleNode::create_subscription<MessageT>(
			topic_name, qos,
			// The move capture "callback = std::move(callback)" is specifically needed
			// here because of the unique_ptr<> in the callback arguments 
			// when the callback is provided with std::bind
			// the returned functor degrades to be move-constructable not copy-constructable
			// https://www.cplusplus.com/reference/functional/bind/
			// using &callback instead can result in a non-deterministic seg fault
			[callback = std::move(callback), this](std::unique_ptr<MessageT> m) 
			{
				try 
				{
					callback(std::move(m));
				} 
				catch (const std::exception &e) 
				{
					handle_primary_state_exception(e);
				}
			},
			options, msg_mem_strat);
	}

	template <typename DurationRepT = int64_t, typename DurationT = std::milli, typename CallbackT>
	std::shared_ptr<rclcpp::TimerBase>
	CMRDVLifecycleNode::create_wall_timer(
		std::chrono::duration<DurationRepT, DurationT> period,
		CallbackT callback,
		rclcpp::CallbackGroup::SharedPtr group) 
	{
		// Change: Local copy of the callback has to be made here because the method is not written to take an rvalue reference
		auto callack_func = [callback = std::move(callback), this]() -> void 
		{
			try 
			{
				callback();
			} 
			catch (const std::exception &e) 
			{
				handle_primary_state_exception(e);
			}
		};

		// create regular ros2 wall timer
		auto timer = rclcpp_lifecycle::LifecycleNode::create_wall_timer(period, callack_func, group);
		// Change: add this timer to list of timers
		timers_.push_back(timer);
		return timer;
	}


	// Change: completly add this new function
	template <typename CallbackT>
	typename rclcpp::TimerBase::SharedPtr
	CMRDVLifecycleNode::create_timer(
		rclcpp::Clock::SharedPtr clock,
		rclcpp::Duration period,
		CallbackT &&callback,
		rclcpp::CallbackGroup::SharedPtr group) 
	{
		auto timer = rclcpp::create_timer(
			this,
			clock,
			period,
			[callback = std::move(callback), this]() -> void 
			{
				try 
				{
					callback();
				} 
				catch (const std::exception &e) 
				{
					handle_primary_state_exception(e);
				}
			},
			group);

		timers_.push_back(timer);
		return timer;
	}
  

	template <typename ServiceT, typename CallbackT>
	typename rclcpp::Service<ServiceT>::SharedPtr
	CMRDVLifecycleNode::create_service(
		const std::string &service_name,
		CallbackT &&callback,
		const rmw_qos_profile_t &qos_profile,
		rclcpp::CallbackGroup::SharedPtr group) 
	{
		// create regular ros2 service with local copy of callback
		return rclcpp_lifecycle::LifecycleNode::create_service<ServiceT>(
			service_name, 
			[callback = std::move(callback), this](std::shared_ptr<rmw_request_id_t> header, std::shared_ptr<typename ServiceT::Request> req, std::shared_ptr<typename ServiceT::Response> resp)
			{
				try 
				{
					callback(header, req, resp);
				} 
				catch (const std::exception &e) 
				{
					handle_primary_state_exception(e);
				}
			},
			qos_profile, group);
	}


	template <class ServiceT>
	typename rclcpp::Client<ServiceT>::SharedPtr
	CMRDVLifecycleNode::create_client (
		const std::string service_name, 
		const rmw_qos_profile_t & qos_profile,
		rclcpp::CallbackGroup::SharedPtr group) 
	{
		// nullptr is the default argument for group
		// when nullptr is provided use the class level service group
		// this is needed instead of a default argument because you cannot change default arguments in overrides
		if (group == nullptr) 
		{ 
			group = this->service_callback_group_;
		}

		return rclcpp_lifecycle::LifecycleNode::create_client<ServiceT>(
			service_name,
			qos_profile,
			group
		); // Change: Our override specifies a different default callback group
	}
}



#endif // CMRDV_NODE_UTILS__CMRDV_LIFECYCLE_NODE_HPP_"