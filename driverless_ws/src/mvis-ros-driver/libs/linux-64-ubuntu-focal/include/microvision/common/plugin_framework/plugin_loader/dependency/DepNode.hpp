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

#include "../LibraryData.hpp"
#include "../dynamic/DynamicLibrary.hpp"

#include <microvision/common/logging/LogManager.hpp>

#include <string>
#include <memory>
#include <utility>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_loader {
//==============================================================================

class DepNode
{
public:
    explicit DepNode(const PluginKey& pluginKey);

    virtual ~DepNode();

public:
    //! Returns true if this node is a descendant of the specific node.
    bool isDescendantOf(std::shared_ptr<DepNode>);

    const PluginKey& getKey() const { return m_key; }

public:
    LibraryDataPtr getLibraryData() const { return m_libraryData; }

    void setLibraryData(LibraryDataPtr libraryData) { m_libraryData = std::move(libraryData); }

public:
    DynamicLibraryPtr getDynamicLibrary() const { return m_dynamicLibrary; }

    void setDynamicLibrary(DynamicLibraryPtr dynamicLibrary) { m_dynamicLibrary = std::move(dynamicLibrary); }

    void removeDynamicLibrary();

public:
    void insertChild(std::shared_ptr<DepNode> childNode);

    const std::vector<std::shared_ptr<DepNode>>& getChildren() const { return m_childNodes; }

public:
    void increaseRefCount() { ++m_refCount; }

    void decreaseRefCount()
    {
        if (m_refCount > 0)
        {
            m_refCount--;
        }
    }

    const uint16_t& getRefCount() const { return m_refCount; }

private:
    logging::LoggerSPtr logger()
    {
        return logging::LogManager::getInstance().createLogger(
            "microvision::common::plugin_framework::plugin_loader::DepNode");
    }

private:
    PluginKey m_key;
    uint16_t m_refCount{0};
    LibraryDataPtr m_libraryData;
    DynamicLibraryPtr m_dynamicLibrary;
    std::vector<std::shared_ptr<DepNode>> m_childNodes;
}; // DepNode

//==============================================================================

using DepNodePtr = std::shared_ptr<DepNode>;

//==============================================================================
} // namespace plugin_loader
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
