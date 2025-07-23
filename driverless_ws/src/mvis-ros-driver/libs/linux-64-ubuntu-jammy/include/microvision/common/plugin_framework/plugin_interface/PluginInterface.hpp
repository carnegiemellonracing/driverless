//==============================================================================
//! \file
//! \brief
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
namespace microvision {
namespace common {
namespace plugin_framework {
namespace plugin_interface {
//==============================================================================

//==============================================================================
//! \brief The base class for all plugin interfaces. Depending on the
//!       plugin type certain derivations can be expected.
//------------------------------------------------------------------------------
class PluginInterface
{
public:
    PluginInterface() = default;

    virtual ~PluginInterface() = default;
}; // PluginInterface

//==============================================================================
} // namespace plugin_interface
} // namespace plugin_framework
} // namespace common
} // namespace microvision
//==============================================================================
