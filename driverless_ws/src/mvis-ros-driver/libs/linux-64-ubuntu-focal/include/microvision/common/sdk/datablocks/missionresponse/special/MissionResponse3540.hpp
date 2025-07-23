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
//! \date Sept 05, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Response to a mission definition for datatype 0x3540.
//------------------------------------------------------------------------------
class MissionResponse3540 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static constexpr uint8_t nbOfReserved = 4;

public: // type declaration
    using ReservedArray = std::array<uint32_t, nbOfReserved>;

public:
    //! Response to a mission specified by m_missionId
    //! This type is serialized as UINT8, 255 is the maximum value
    enum class Response : uint8_t
    {
        Accepted     = 0,
        Accomplished = 1,
        Aborted      = 2,
        Denied       = 3,
        Error        = 255
    };

public:
    MissionResponse3540();
    virtual ~MissionResponse3540() = default;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.missionresponse3540"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    void setTimestamp(const NtpTime& timestamp) { this->m_timestamp = timestamp; }
    void setResponse(const Response response) { this->m_response = response; }
    void setMissionId(const uint32_t missionId) { this->m_missionId = missionId; }

public:
    const NtpTime& getTimestamp() const { return m_timestamp; }
    Response getResponse() const { return m_response; }
    uint32_t getMissionId() const { return m_missionId; }

    //==============================================================================
    //! \brief Returns value from m_reserved array at specified position \a idx.
    //! \param[in] idx  Index specifying position in m_reserved array.
    //! \return Value of m_reserved array at position \a idx. Returns last
    //!         element of array, if \a idx >= \a nbOfReserved.
    //------------------------------------------------------------------------------
    uint32_t getReserved(const uint_fast8_t idx) const;

protected:
    NtpTime m_timestamp{}; //!< Timestamp of response message
    Response m_response{Response::Denied}; //!< Response to a received mission definition/request
    uint32_t m_missionId{0}; //!< Id of mission, which triggered this response

private:
    ReservedArray m_reserved{{}};

}; // MissionResponse3540

//==============================================================================

bool operator==(const MissionResponse3540& lhs, const MissionResponse3540& rhs);
bool operator!=(const MissionResponse3540& lhs, const MissionResponse3540& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
