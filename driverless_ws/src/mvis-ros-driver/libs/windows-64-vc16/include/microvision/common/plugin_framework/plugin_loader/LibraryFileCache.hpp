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

#include <microvision/common/logging/logging.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_loader {
//==============================================================================

class LibraryFileCache
{
public:
    explicit LibraryFileCache(const std::vector<std::string>& libraryFolderList);

public:
    void addLibrary(LibraryDataPtr libraryData);

    void addFolder(std::string folderPath);

    void rebuildFolder(std::string folderPath);

    void rebuildAll();

    const std::vector<LibraryDataPtr>& getCachedLibraryDataList() const { return m_libraryDataList; };

private:
    void loadCache(const std::string& folderPath);

    void initCache(const std::string& folderPath);

    bool existFileCache(const std::string& folderPath);

    bool existFolderInList(const std::string& folderPath);

    void openFileStreamForRead(const std::string& folderPath);

    void openFileStreamForWrite(const std::string& folderPath, const bool clearFile = false, const bool append = false);

    void closeFileStream();

    void addApiVersionLabel();

    const std::string getRegFilePath(const std::string& folderPath) const
    {
        return std::string(folderPath).append("registeredLibraries.txt");
    }

    bool checkApiVersion(const std::string& apiVersion);

    //-------------------------------------------------------------------------

    logging::LoggerSPtr logger()
    {
        return logging::LogManager::getInstance().createLogger(
            "microvision::common::plugin_framework::plugin_loader::LibraryFileCache");
    }

private:
    std::fstream m_regFileStream{};
    std::vector<std::string> m_libraryFolderList{};
    std::vector<LibraryDataPtr> m_libraryDataList{};
}; // LibraryFileCache

//==============================================================================
} // namespace plugin_loader
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
