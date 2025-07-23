//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) Microvision 2010-2024
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! MicroVisionLicense.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//==============================================================================

#pragma once

//==============================================================================

#include "DepNode.hpp"

#include <microvision/common/logging/LogManager.hpp>

#include <string>
#include <memory>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_loader {
//==============================================================================

class DepGraph
{
public:
    virtual ~DepGraph();

public:
    void insert(LibraryDataPtr libraryData);

    bool containsKey(const PluginKey& pluginKey) const { return getNode(pluginKey) != nullptr; }

public:
    bool containsCycle();

    bool containsCycle(DepNodePtr node);

public:
    const std::vector<DepNodePtr>& getNodes() const { return m_nodes; };

    DepNodePtr getNode(const PluginKey& pluginKey) const;

private:
    void addNode(DepNodePtr node);

    logging::LoggerSPtr logger()
    {
        return logging::LogManager::getInstance().createLogger(
            "microvision::common::plugin_framework::plugin_loader::DepGraph");
    }

private:
    std::vector<DepNodePtr> m_nodes;
}; // DepGraph

//==============================================================================
} // namespace plugin_loader
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
