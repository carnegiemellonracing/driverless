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
#include "LibraryFileCache.hpp"
#include "LibraryData.hpp"

#include <microvision/common/plugin_framework/plugin_interface/PluginApi.hpp>
#include <microvision/common/plugin_framework/plugin_loader/dynamic/DynamicLibrary.hpp>
#include <microvision/common/plugin_framework/plugin_loader/dependency/DepGraph.hpp>

#include <microvision/common/logging/LogManager.hpp>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dirent.h>
#endif

#include <map>
#include <regex>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_loader {
//==============================================================================

class LibraryRegistry
{
public:
    using ConstCString = const char*;

public:
    explicit LibraryRegistry(const PluginLoaderConfig& config);

public:
    void registerLibraries(std::string folderPath);

    void registerLibraries(const std::vector<std::string>& libraryFolderList);

public:
    void registerLibrary(std::string folderPath, const std::string& filename);

    void registerLibrary(const std::string& folderPath,
                         const std::string& filename,
                         const plugin_interface::MetaData& metaData,
                         const std::string& extendedMetaData);

public:
    DynamicLibraryPtr loadLibrary(const std::string& pluginName, const std::string& version);

    void unloadLibrary(DynamicLibraryPtr library);

    bool isLoaded(const std::string& pluginName, const std::string& version) const;

public:
    void addLibraryFolder(const std::string& folderPath);

    void rebuildRegistry(const std::vector<std::string>& libraryFolderList = {});

public:
    std::vector<LibraryDataPtr> getLibraryDataList() const;

public:
    void getApiVersion(DynamicLibraryPtr library, uint32_t*& apiVersion) const;
    void getApiType(DynamicLibraryPtr library, ConstCString& apiType) const
    {
        getStringData(library, apiType, "apiType");
    }

    void getMetaData(DynamicLibraryPtr library, plugin_interface::MetaData*& metaData) const;
    void getExtendedMetaData(DynamicLibraryPtr library, ConstCString& extendedMetaData) const
    {
        getStringData(library, extendedMetaData, "extendedMetaData");
    }

public:
    logging::LoggerSPtr logger()
    {
        return logging::LogManager::getInstance().createLogger(
            "microvision::common::plugin_framework::plugin_loader::LibraryRegistry");
    }

private:
    bool checkInfo(DynamicLibraryPtr lib,
                   uint32_t& apiVersion,
                   ConstCString& apiType,
                   plugin_interface::MetaData*& metaData,
                   ConstCString& extendedMetaData,
                   const std::string& folderPath,
                   const std::string& filename);

    void getStringData(DynamicLibraryPtr lib, ConstCString& str, const char* const symbolName) const;

    DynamicLibraryPtr openLibrary(const std::string& folderPath, const std::string& filename);

    DepNodePtr loadChildLibraries(const DepNodePtr& node);

    DynamicLibraryPtr getLibraryByFolderPathAndFilename(const std::string& folderPath,
                                                        const std::string& filename) const;

    LibraryDataPtr getLibraryData(const std::string& pluginName, const std::string& version);

    bool isRegistered(const plugin_interface::MetaData& metaData);

    bool hasValidExtension(const std::string& fileName) const;

    //========================================
    //! \brief Checks, whether \a fileName is not matching \a namePattern.
    //!
    //! Not matching means, \a namePattern excludes \a fileName.
    //!
    //! \param[in] namePattern  A regular expression providing the name filter.
    //! \param[in] fileName     File name to be checked whether it is excluded
    //!                         by \a namePattern.
    //! \return \c true if fileName is excluded by \a namePattern, \c false otherwise.
    //----------------------------------------
    bool isNameExcluded(const std::regex& namePattern, const std::string& fileName) const;

private:
    static const constexpr char* const supportedApiType{"microvision.plugin"}; //!< Supported plugin type name
    static constexpr std::size_t supportedApiTypeLength{19}; //!< Supported plugin type name length

private:
    PluginLoaderConfig m_config;
    DepGraph m_dependencyGraph{};
    std::shared_ptr<LibraryFileCache> m_libraryFileCache{};
}; // LibraryRegistry

//==============================================================================
} // namespace plugin_loader
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
