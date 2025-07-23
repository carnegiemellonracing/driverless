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

#include <functional>
#include <regex>
#include <string>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_loader {
//==============================================================================

//==============================================================================
//! \brief A function type to check the extendedMetaData to be acceptable.
//!
//! The first parameter are the extendedMetaData. The second is the folder name
//! of the library and the third one the filename of the library. The names are
//! needed for error message output only.
//------------------------------------------------------------------------------
using ExtendedMetaDataCheck = std::function<bool(const std::string&, const std::string&, const std::string&)>;

//==============================================================================

struct PluginLoaderConfig
{
    std::vector<std::string> libraryFolderList{};
    std::string typeInMetaData{};
    ExtendedMetaDataCheck extendedMetaDataCheck;
    std::regex namePattern{".*"}; //!< A name of a plugin candidate has to match the regex.
    bool cacheLibraries{false};
}; // PluginLoaderConfig

//==============================================================================
//! \brief Creates a regex that excludes plugin names beginning with one of the strings
//!        out of \a excludedNames.
//! \param[in] excludedNames  Plugins names beginning with one of these names shall
//!                           be excluded.
//! \return A regex that does not match with any plugin name that starts with one of the
//!         strings from \a excludedNames.
//------------------------------------------------------------------------------
std::regex createRegExExcluding(const std::vector<std::string>& excludedNames);

//==============================================================================
//! \brief Creates a regex that includes plugin names containing one of the strings
//!        out of \a allowedNames.
//! \param[in] allowedNames  Plugins names containing one of these strings shall
//!                          be included.
//! \return A regex that does match with any plugin name that contains one of the
//!         strings from \a allowedNames.
//------------------------------------------------------------------------------
std::regex createRegExIncluding(const std::vector<std::string>& allowedNames);

//==============================================================================
//! \brief Creates a regex that excludes plugin names beginning with one string out of
//!        \a excludedNames and only include plugin names containing a string out of
//!        \a allowedNames.
//!
//! First the exclude part will be applied, if the tested plugin name passes this, the
//! include part will be applied for a final check.
//!
//! \param[in] excludedNames  Plugins names beginning with one of these names shall
//!                           be excluded.
//! \param[in] allowedNames   Plugins names containing one of these names shall
//!                           be included.
//!
//! \return A regex that does not match with any plugin name that starts with one of the
//!         strings from \a excludedNames \b and does match with any plugin name that
//!         contains one of the strings from \a allowedNames.
//------------------------------------------------------------------------------
std::regex createRegEx(const std::vector<std::string>& excludedNames, const std::vector<std::string>& allowedNames);

//==============================================================================
} // namespace plugin_loader
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
