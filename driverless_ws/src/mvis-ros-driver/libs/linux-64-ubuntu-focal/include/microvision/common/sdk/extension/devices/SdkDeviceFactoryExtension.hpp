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
//! \date Feb 17, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/extension/DeviceFactoryExtension.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Factory interface for extension devices.
//!
//! This interface will create plugin devices.
//!
//! \extends DeviceFactoryExtension
//------------------------------------------------------------------------------
class SdkDeviceFactoryExtension : public DeviceFactoryExtension
{
public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~SdkDeviceFactoryExtension() override = default;

public:
    //========================================
    //! \brief Get list of type names for devices that this extension can create.
    //!
    //! \return Type names of devices that this extension can create.
    //----------------------------------------
    const std::vector<std::string>& getTypeNames() const override;

    //========================================
    //! \brief Create a device from type name.
    //! \param[in] deviceTypeName  Device type name.
    //! \return Pointer to new device instance created by this factory extension.
    //----------------------------------------
    DevicePtr createFromTypeName(const std::string& deviceTypeName) const override;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
