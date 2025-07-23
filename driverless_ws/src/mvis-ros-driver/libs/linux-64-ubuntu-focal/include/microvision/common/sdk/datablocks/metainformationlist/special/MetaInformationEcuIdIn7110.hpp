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

#include <string>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Meta information list entry type for Ecu Id.
//==============================================================================
class MetaInformationEcuIdIn7110 final : public MetaInformationBaseIn7110
{
public:
    //========================================
    //! \brief Default Constructor
    //----------------------------------------
    MetaInformationEcuIdIn7110() : MetaInformationBaseIn7110(MetaInformationBaseIn7110::MetaInformationType::EcuId) {}

    //========================================
    //! \brief Default Destructor
    //----------------------------------------
    virtual ~MetaInformationEcuIdIn7110() = default;

public:
    //========================================
    //! \brief get the ecu id stored in this measurement list entry.
    //!
    //! \return  Ecu id string.
    //----------------------------------------
    const std::string& getEcuId() const { return m_ecuId; }

    //========================================
    //! \brief Set the new ecu id.
    //!
    //! \param[in] newEcuId  String ecu id to store in this list entry.
    //----------------------------------------
    void setEcuId(const std::string& newEcuId);

public:
    //========================================
    //! \brief Compare meta information list entry with another one.
    //!
    //! \param[in] otherBase  Entry to compare with.
    //! \return \c True if equal, \c false if not.
    //----------------------------------------
    bool isEqual(const MetaInformationBaseIn7110& otherBase) const override;

    //========================================
    //! \brief Return the serialized size of this entry.
    //!
    //! \return The size in bytes.
    //----------------------------------------
    uint32_t getSerializedPayloadSize() const override;

    //========================================
    //! \brief Deserialize binary data from stream into this meta information list entry.
    //!
    //! \param[in, out] is      Stream to read from.
    //! \param[in] payloadSize  Size of serialized payload.
    //! \return  \c true if success, \c false if not.
    //----------------------------------------
    bool deserializePayload(std::istream& is, const uint32_t payloadSize) override;

    //========================================
    //! \brief Serialize this meta information list entry into binary data from stream.
    //!
    //! \param[in, out] os  Stream to write to.
    //! \return  \c true if success, \c false if not.
    //----------------------------------------
    bool serializePayload(std::ostream& os) const override;

private:
    std::string m_ecuId{};
}; // MetaInformationEcuIdIn7110

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
