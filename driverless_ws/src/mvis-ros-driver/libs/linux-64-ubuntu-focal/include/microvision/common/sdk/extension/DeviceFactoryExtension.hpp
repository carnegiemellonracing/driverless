//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 12, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/Device.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface for the plugin device extension
//!
//! This interface will create plugin devices.
//------------------------------------------------------------------------------
class DeviceFactoryExtension
{
public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~DeviceFactoryExtension() = default;

public:
    //========================================
    //! \brief Get list of type names for devices that this factory can create.
    //!
    //! \return Type names of devices that this factory can create.
    //----------------------------------------
    virtual const std::vector<std::string>& getTypeNames() const = 0;

    //========================================
    //! \brief Create a device of the given type name.
    //! \param[in] deviceTypeName  Device type name.
    //! \return Device created by this factory.
    //----------------------------------------
    virtual DevicePtr createFromTypeName(const std::string& deviceTypeName) const = 0;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
