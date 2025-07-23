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
//! \date 02.November 2018
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
//! \brief Current mission status.
//------------------------------------------------------------------------------
class MissionHandlingStatus3530 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //! \enum State
    //! State of mission handling module
    //! This type is serialized as UINT8, 255 is the maximum value
    // ----------------------------------------
    enum class State : uint8_t
    {
        Idle              = 0,
        MissionInProgress = 1,
        NotReady          = 255
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.missionhandlingstatus3530"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    static constexpr uint8_t nbOfReserved = 4;

public:
    using ArrayOfReserved = std::array<uint32_t, nbOfReserved>;

public:
    MissionHandlingStatus3530();
    virtual ~MissionHandlingStatus3530() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // setters and getters
    void setTimestamp(const NtpTime& timestamp) { this->m_timestamp = timestamp; }
    void setState(const State state) { this->m_state = state; }
    void setMissionId(const uint32_t missionId) { this->m_missionId = missionId; }

    const NtpTime& getTimestamp() const { return m_timestamp; }
    State getState() const { return m_state; }
    uint32_t getMissionId() const { return m_missionId; }
    ArrayOfReserved getReserved() const& { return m_reserved; }

protected:
    NtpTime m_timestamp{}; //!< Timestamp of status message
    State m_state{State::NotReady}; //!< Indicates current state of mission handling module
    uint32_t m_missionId{0}; //!< Id of current mission

private:
    ArrayOfReserved m_reserved{{}};

}; //MissionHandlingStatus3530

//==============================================================================

bool operator==(const MissionHandlingStatus3530& lhs, const MissionHandlingStatus3530& rhs);
bool operator!=(const MissionHandlingStatus3530& lhs, const MissionHandlingStatus3530& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
