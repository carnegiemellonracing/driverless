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
//! A microvision plugin API contains the following symbols.
//! \code
//! extern "C" {
//! extern DLL_EXPORT microvision::common::plugin_framework::plugin_interface::PluginInterface* getInstance();
//! extern DLL_EXPORT microvision::common::plugin_framework::plugin_interface::MetaData metaData;
//! extern DLL_EXPORT const char* extendedMetaData;
//!
//! extern DLL_EXPORT const uint32_t apiVersion;
//! extern DLL_EXPORT const char* apiType;
//! } // extern "C"
//! \endcode
//! Where DLL_EXPORT is the export symbol for this library/plugin.
//!
//! With the values for apiVersion and apiType
//! \code
//! const uint32_t apiVersion{microvision::common::plugin_framework::plugin_interface::pluginApiVersion};
//! const char* apiType{"microvision.plugin"};
//! \endcode
//!
//! For pluginApiVersion see below.
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include "PluginInterface.hpp"

#ifdef _WIN32
#    include <windows.h>
#endif

#include <cstdint>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_interface {
//==============================================================================

// Define the supported API version
constexpr uint32_t pluginApiVersion{1};
constexpr const char* pluginApiVersionStr{"API version"};

//==============================================================================
#ifdef _WIN32
struct lib_t
{
    HMODULE handle;
    char* errorMessage;
};
#else
struct lib_t
{
    void* handle;
    char* errorMessage;
};
#endif

//==============================================================================

// Define a type for the static function pointer.
using GetPluginFunc = PluginInterface* (*)(void);

//==============================================================================

//==============================================================================
//! \brief A plugin ID consists out of the plugin's name and version.
//------------------------------------------------------------------------------
struct PluginId
{
    const char* name;
    const char* version;
};

//========================================
//! \brief A plugin can depend on several other plugins.
//!        Each identified by its PluginId.
//----------------------------------------
struct PluginDependencies
{
    const uint16_t nbOfEntries;
    const PluginId* const entries;
};

//========================================
//! \brief All plugin informations combined.
//----------------------------------------
struct PluginDetails
{
    const PluginId id; //!< The id of this plugin.
    const char* type; //!< Plugin Type.
    const char* description; //!< Plugin description.
    const PluginDependencies dependencies; //!< The plugins this plugin depends on.
};

//========================================
//! \brief An instance called metaData of this struct
//!       is part of the plugin's API.
//----------------------------------------
struct MetaData
{
    const PluginDetails pluginDetails;
};

//==============================================================================
} // namespace plugin_interface
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
