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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_loader {
//==============================================================================

struct PluginKey
{
public:
    std::string name;
    std::string version;
}; // PluginKey

//==============================================================================

using PluginKeyPtr = std::shared_ptr<PluginKey>;

//==============================================================================

std::string serialize(const PluginKey& pk);
void deserialize(PluginKey& pk, const std::string& serialized);

//==============================================================================

inline bool operator==(const PluginKey& lhs, const PluginKey& rhs)
{
    return (lhs.name == rhs.name) && (lhs.version == rhs.version);
}

//==============================================================================
//==============================================================================
//==============================================================================

struct PluginInfo
{
public:
    PluginKey pluginKey{};
    std::string type;
    std::string description;
    std::string extendedMetaData;
    std::vector<PluginKeyPtr> dependencies{};
}; // PluginInfo

//==============================================================================

std::string serialize(const PluginInfo& pi);
void deserialize(PluginInfo& pi, const std::string& serialized);

//==============================================================================
//==============================================================================
//==============================================================================

struct LibraryData
{
public:
    PluginInfo pluginInfo;
    std::string folder;
    std::string filename;
}; // LibraryData

//==============================================================================

std::string serialize(const LibraryData& ld);
void deserialize(LibraryData& ld, const std::string& serialized);

//==============================================================================

using LibraryDataPtr = std::shared_ptr<LibraryData>;

//==============================================================================
} // namespace plugin_loader
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
