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

#include "LibraryData.hpp"

#include <microvision/common/plugin_framework/plugin_interface/PluginApi.hpp>

#include <string>
#include <vector>
#include <sstream>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_loader {
//==============================================================================

class PluginLoaderUtil
{
public:
    static inline const std::string generateKey(const LibraryData& libraryData)
    {
        return generateKey(libraryData.pluginInfo.pluginKey);
    }

    static inline const std::string generateKey(const PluginKey& key)
    {
        return std::string(key.name).append(key.version);
    }

public:
    static LibraryDataPtr createLibraryData(const std::string& folderPath,
                                            const std::string& filename,
                                            const plugin_interface::MetaData& metaData,
                                            const std::string& extendedMetaData);

public:
    static void appendSeparator(std::string& folderPath);
    static char pathSeparator();
    static std::string getExtension();
}; // PluginLoaderUtil

//==============================================================================
} // namespace plugin_loader
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
