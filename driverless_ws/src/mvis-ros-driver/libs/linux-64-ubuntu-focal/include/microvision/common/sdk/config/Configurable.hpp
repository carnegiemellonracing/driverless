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
//! \date Nov 5, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <string>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface to mark an inheritance as configurable.
//!
//! For example this can be used for device or impoerter configuration.
//!
//! \sa ImporterBase
//! \sa Device
//------------------------------------------------------------------------------
class Configurable
{
public:
    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~Configurable() = default;

public:
    //========================================
    //! \brief Get supported types of configuration.
    //!
    //! Configuration type is a human readable unique string name of the configuration
    //! used to address it in code.
    //!
    //! \return All supported configuration types.
    //----------------------------------------
    virtual const std::vector<std::string>& getConfigurationTypes() const = 0;

    //========================================
    //! \brief Get whether a configuration is mandatory for this Configurable.
    //! \return \c true if a configuration is mandatory for this Configurable,
    //!         \c false otherwise.
    //----------------------------------------
    virtual bool isConfigurationMandatory() const = 0;
}; // class Configurable

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
