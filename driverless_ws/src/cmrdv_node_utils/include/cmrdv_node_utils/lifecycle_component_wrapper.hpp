
#ifndef CMRDV_NODE_UTILS__LIFECYCLE_COMPONENT_WRAPPER_HPP__
#define CMRDV_NODE_UTILS__LIFECYCLE_COMPONENT_WRAPPER_HPP__


#include <map>
#include <unordered_map>
#include <tuple>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/optional.hpp>

#include "composition_interfaces/srv/load_node.hpp"
#include "composition_interfaces/srv/unload_node.hpp"
#include "composition_interfaces/srv/list_nodes.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executor.hpp"
#include "rclcpp/node_options.hpp"
#include <rclcpp_components/component_manager.hpp>

#include "rclcpp_components/node_factory.hpp"
#include "rclcpp_components/visibility_control.hpp"

#include "cmrdv_node_utils/cmrdv_lifecycle_node.hpp"



namespace class_loader 
{
    class ClassLoader;
}  // namespace class_loader


namespace cmrdv_ros2_utils 
{

    /// LifecycleComponentWrapper handles the services to load, unload, and get the list of loaded components.
    class LifecycleComponentWrapper : public cmrdv_node_utils::CMRDVLifecycleNode 
    {

        public:

            using LoadNode = composition_interfaces::srv::LoadNode;
            using UnloadNode = composition_interfaces::srv::UnloadNode;
            using ListNodes = composition_interfaces::srv::ListNodes;


            /// Represents a component resource.
            /**
             * Is a pair of class name (for class loader) and library path (absolute)
             */
            using ComponentResource = std::pair<std::string, std::string>;


            /// Default constructor
            /**
             * Initializes the component manager. It creates the services: load node, unload node
             * and list nodes.
             *
             * \param node_options additional options to control creation of the node.
             */
            RCLCPP_COMPONENTS_PUBLIC
            LifecycleComponentWrapper(
                const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions()
                    .start_parameter_services(false)
                    .start_parameter_event_publisher(false));

            
            /**
             * \brief Method to initialize the component management services.
             *        NOTE: It is critical to call this method before .spin()
             * 
             * The reason this method is not part of the constructor is to conform to the CMRDVLifecycleNode API.
             * 
             * \param executor the executor which will spin the node.
             */ 
            RCLCPP_COMPONENTS_PUBLIC
            void initialize(std::weak_ptr<rclcpp::Executor> executor);


            RCLCPP_COMPONENTS_PUBLIC
            virtual ~LifecycleComponentWrapper();


            /// Return a list of valid loadable components in a given package.
            /**
             * \param package_name name of the package
             * \param resource_index name of the executable
             * \throws ComponentManagerException if the resource was not found or a invalid resource entry
             * \return a list of component resources
             */
            RCLCPP_COMPONENTS_PUBLIC
            virtual std::vector<ComponentResource>
            get_component_resources(
                const std::string & package_name,
                const std::string & resource_index = "rclcpp_components") const;


            /// Instantiate a component from a dynamic library.
            /**
             * \param resource a component resource (class name + library path)
             * \return a NodeFactory interface
             */
            RCLCPP_COMPONENTS_PUBLIC
            virtual std::shared_ptr<rclcpp_components::NodeFactory>
            create_component_factory(const ComponentResource & resource);


        protected:

            /// Create node options for loaded component
            /**
             * \param request information with the node to load
             * \return node options
             */
            RCLCPP_COMPONENTS_PUBLIC
            virtual rclcpp::NodeOptions create_node_options(
                const std::shared_ptr<LoadNode::Request> request);
    

            /// Service callback to load a new node in the component
            /**
             * This function allows to add parameters, remap rules, a specific node, name a namespace
             * and/or additional arguments.
             *
             * \param request_header unused
             * \param request information with the node to load
             * \param response
             * 
             * 
             * ///// CMRDV CHANGE /////
             * \param internal_call Optional parameter which if set means that the node should be loaded with the specified unique_id. 
             *                      This is used internally to track deferred node loading.  
             * ///// END CMRDV CHANGE /////
             * 
             * 
             * \throws std::overflow_error if node_id suffers an overflow. Very unlikely to happen at 1 kHz
             *   (very optimistic rate). it would take 585 years.
             * \throws ComponentManagerException In the case that the component constructor throws an
             *   exception, rethrow into the following catch block.
             */
            RCLCPP_COMPONENTS_PUBLIC
            virtual void on_load_node(
                const std::shared_ptr<rmw_request_id_t> request_header,
                const std::shared_ptr<LoadNode::Request> request,
                std::shared_ptr<LoadNode::Response> response);


            /// Service callback to unload a node in the component
            /**
             * \param request_header unused
             * \param request unique identifier to remove from the component
             * \param response true on the success field if the node unload was succefully, otherwise false
             *   and the error_message field contains the error.
             */
            RCLCPP_COMPONENTS_PUBLIC
            virtual void on_unload_node(
                const std::shared_ptr<rmw_request_id_t> request_header,
                const std::shared_ptr<UnloadNode::Request> request,
                std::shared_ptr<UnloadNode::Response> response);


            /// Service callback to get the list of nodes in the component
            /**
             * Return a two list: one with the unique identifiers and other with full name of the nodes.
             *
             * \param request_header unused
             * \param request unused
             * \param response list with the unique ids and full node names
             */
            RCLCPP_COMPONENTS_PUBLIC
            virtual void on_list_nodes(
                const std::shared_ptr<rmw_request_id_t> request_header,
                const std::shared_ptr<ListNodes::Request> request,
                std::shared_ptr<ListNodes::Response> response);


            ///// Overrides /////

            // handle_on_configure is not used
            cmrdv_ros2_utils::CallbackReturn handle_on_activate(const rclcpp_lifecycle::State &prev_state) override;
            cmrdv_ros2_utils::CallbackReturn handle_on_deactivate(const rclcpp_lifecycle::State &prev_state) override;
            cmrdv_ros2_utils::CallbackReturn handle_on_cleanup(const rclcpp_lifecycle::State &prev_state) override;
            cmrdv_ros2_utils::CallbackReturn handle_on_error(const rclcpp_lifecycle::State &prev_state, const std::string &exception_string) override;
            cmrdv_ros2_utils::CallbackReturn handle_on_shutdown(const rclcpp_lifecycle::State &prev_state) override;

            ///// END CMRDV CHANGE /////

        private:

            std::weak_ptr<rclcpp::Executor> executor_;

            uint64_t unique_id_ {1};
            std::map<std::string, std::unique_ptr<class_loader::ClassLoader>> loaders_;
            std::map<uint64_t, rclcpp_components::NodeInstanceWrapper> node_wrappers_;

            rclcpp::Service<LoadNode>::SharedPtr loadNode_srv_;
            rclcpp::Service<UnloadNode>::SharedPtr unloadNode_srv_;
            rclcpp::Service<ListNodes>::SharedPtr listNodes_srv_;

            //// CMRDV CHANGE ////
            
            //! A record of all the node load requests organized by their unique id
            std::unordered_map<uint64_t, std::pair<rmw_request_id_t, LoadNode::Request>> load_node_requests_;

            //// END CMRDV CHANGE ////

    };

}  // namespace rclcpp_components

#endif // CMRDV_NODE_UTILS__LIFECYCLE_COMPONENT_WRAPPER_HPP__

