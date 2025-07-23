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
//! \date Mar 02, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/DeviceConfiguration.hpp>
#include <microvision/common/sdk/config/io/TcpConfiguration.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Configuration for the MVIS ECU
//------------------------------------------------------------------------------
class MvisEcuConfiguration : public DeviceConfiguration, public TcpConfiguration
{
public:
    // ==================================
    //! \brief Configuration type name
    //----------------------------------------
    static MICROVISION_SDK_API const std::string typeName;

    //========================================
    //! \brief Unique config id for property 'data type ranges'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string dataTypeRangesConfigId;

public:
    //========================================
    //!\brief A pair of DataTypeIds. Describing a range.
    //----------------------------------------
    using DataTypeRange = std::pair<DataTypeId, DataTypeId>;

    //========================================
    //!\brief A vector of Ranges.
    //----------------------------------------
    using DataTypeRangeVector = std::vector<DataTypeRange>;

public:
    //========================================
    //! \brief Empty constructor
    //----------------------------------------
    MvisEcuConfiguration();

    //========================================
    //! \brief Copy constructor
    //----------------------------------------
    MvisEcuConfiguration(const MvisEcuConfiguration& other);

    //========================================
    //! \brief Disable move constructor to ensure thread safety
    //----------------------------------------
    MvisEcuConfiguration(MvisEcuConfiguration&&) = delete;

    //========================================
    //! \brief Destructor
    //----------------------------------------
    ~MvisEcuConfiguration() override;

public:
    //========================================
    //! \brief Return the configuration type
    //! \returns Configuration type
    //----------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Get copy of configuration
    //! \returns Pointer to newly copied configuration
    //----------------------------------------
    ConfigurationPtr copy() const override;

public:
    //========================================
    //! \brief Get the data types ranges which shall be filtered
    //! \returns Filtered data type ranges
    //----------------------------------------
    ConfigurationPropertyOfType<DataTypeRangeVector>& getDataTypeRanges();

private:
    //========================================
    //! \brief Data type ranges property
    //----------------------------------------
    ConfigurationPropertyOfType<DataTypeRangeVector> m_dataTypeRanges;
};

//=================================================
//! \brief Nullabe MvisEcuConfiguration pointer
//-------------------------------------------------
using MvisEcuConfigurationPtr = std::shared_ptr<MvisEcuConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
