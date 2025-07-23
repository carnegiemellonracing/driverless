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
//! \date Apr 5, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationBaseIn7110.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class MetaInformationAppBaseConfigIn7110 final : public MetaInformationBaseIn7110
{
public:
    MetaInformationAppBaseConfigIn7110()
      : MetaInformationBaseIn7110(MetaInformationBaseIn7110::MetaInformationType::AppBaseConfig)
    {}
    virtual ~MetaInformationAppBaseConfigIn7110() = default;

public:
    const std::string& getAppBaseConfig() const { return m_appBaseConfig; }
    void setAppBaseConfig(const std::string& newAppBaseConfig);

public:
    bool isEqual(const MetaInformationBaseIn7110& otherBase) const override;
    uint32_t getSerializedPayloadSize() const override;
    bool deserializePayload(std::istream& is, const uint32_t payloadSize) override;
    bool serializePayload(std::ostream& os) const override;

private:
    std::string m_appBaseConfig{}; //!< The AppBase configuration.
}; // MetaInformationAppBaseConfigIn7110

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
