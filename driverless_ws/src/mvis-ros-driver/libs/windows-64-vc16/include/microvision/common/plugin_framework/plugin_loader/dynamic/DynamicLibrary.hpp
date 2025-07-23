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

#include <microvision/common/plugin_framework/plugin_interface/PluginApi.hpp>

#include <string>
#include <stdexcept>
#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_loader {
//==============================================================================

class DynamicLibrary
{
public:
    //! Opens a shared library.
    //! The filename is in utf-8. Returns true on success and false on error.
    //! Call `SharedLibrary::error()` to get the error message.
    bool open(const std::string& path);

    //! Closes the shared library.
    void close();

    //! Retrieves a data pointer from a dynamic library.
    //! It is legal for a symbol to map to nullptr.
    //! Returns 0 on success and -1 if the symbol was not found.
    bool getDataPointerBySymbol(const char* name, void*& ptr);

private:
    void setError(const std::string& prefix);

protected:
    plugin_interface::lib_t m_lib;
    std::string m_error;
}; // DynamicLibrary

//==============================================================================

using DynamicLibraryPtr = std::shared_ptr<DynamicLibrary>;

//==============================================================================
} // namespace plugin_loader
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
