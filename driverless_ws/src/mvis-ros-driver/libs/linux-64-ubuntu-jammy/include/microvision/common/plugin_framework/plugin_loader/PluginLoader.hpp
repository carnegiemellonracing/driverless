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

#include "PluginLoaderConfig.hpp"
#include "LibraryRegistry.hpp"

#include <microvision/common/plugin_framework/plugin_interface/PluginApi.hpp>
#include <microvision/common/plugin_framework/plugin_interface/PluginInterface.hpp>

#include <microvision/common/plugin_framework/plugin_loader/dynamic/DynamicLibrary.hpp>

#include <microvision/common/logging/LogManager.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_loader {
//==============================================================================

class PluginLoader
{
public:
    using ConstCString = const char*;

public:
    explicit PluginLoader(const PluginLoaderConfig& config);

public:
    DynamicLibraryPtr loadLibrary(const std::string& pluginName, const std::string& version);

    void unloadLibrary(const DynamicLibraryPtr& library);

    void unloadLibrary(const std::string& pluginName, const std::string& version);

    bool isLoaded(const std::string& pluginName, const std::string& version) const;

public:
    plugin_interface::PluginInterface* loadPlugin(const std::string& pluginName, const std::string& version);

    plugin_interface::PluginInterface* getPlugin(const DynamicLibraryPtr& library);

    void loadAllPlugins();

public:
    void addLibraryFolder(const std::string& libraryFolder);

    void rebuildRegistry(const std::vector<std::string>& libraryFolderList = {});

public:
    void getMetaData(const DynamicLibraryPtr& library, plugin_interface::MetaData*& metaData) const;

    void getExtendedMetaData(const DynamicLibraryPtr& library, ConstCString& extendedMetaData) const;

public:
    std::vector<LibraryDataPtr> getLibraryDataList() const;

private:
    logging::LoggerSPtr logger()
    {
        return logging::LogManager::getInstance().createLogger(
            "microvision::common::plugin_framework::plugin_loader::PluginLoader");
    }

private:
    LibraryRegistry m_libraryRegistry;
}; // PluginLoader

//==============================================================================
} // namespace plugin_loader
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
