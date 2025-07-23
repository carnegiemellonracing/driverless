//==============================================================================
//! \file
//!
//! \brief Configuration for Scan2340Importer2342.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Oct 12, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/ConfigurationPropertyOfType.hpp>
#include <microvision/common/sdk/config/Configuration.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Configuration for the Scan2340Importer2342.
//------------------------------------------------------------------------------
class Scan2340Importer2342Configuration final : public virtual Configuration
{
public:
    //========================================
    //! \brief Configuration type name.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //==============================================================================
    //! \brief Unique config id for property of 'skip-invalid-points'.
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API const std::string skipInvalidPointsConfigId;

public:
    //========================================
    //! \brief Get name of type of this configuration.
    //! \returns Configuration type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getTypeName();

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Scan2340Importer2342Configuration();

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other Scan2340Importer2342Configuration to copy from.
    //----------------------------------------
    Scan2340Importer2342Configuration(const Scan2340Importer2342Configuration& other);

    //========================================
    //! \brief Disabled move constructor to ensure thread-safety.
    //----------------------------------------
    Scan2340Importer2342Configuration(Scan2340Importer2342Configuration&&) = delete;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~Scan2340Importer2342Configuration() override;

public: // implements Configuration
    //========================================
    //! \brief Get type of configuration to match with.
    //! \returns Configuration type.
    //----------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Get copy of configuration.
    //! \returns Pointer to new copied Configuration.
    //----------------------------------------
    ConfigurationPtr copy() const override;

public:
    //========================================
    //! \brief Get skip invalid points configuration property.
    //! \returns Skip invalid points configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<bool>& getSkipInvalidPoints();

private:
    //========================================
    //! \brief Skip invalid points property.
    //----------------------------------------
    ConfigurationPropertyOfType<bool> m_skipInvalidPoints;
}; // class Scan2340Importer2342Configuration

//==============================================================================
//! \brief Nullable Scan2340Importer2342Configuration pointer.
//------------------------------------------------------------------------------
using Scan2340Importer2342ConfigurationPtr = std::shared_ptr<Scan2340Importer2342Configuration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
