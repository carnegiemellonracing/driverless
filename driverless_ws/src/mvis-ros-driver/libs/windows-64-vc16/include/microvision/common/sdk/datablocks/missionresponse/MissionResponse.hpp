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
//! \date May 27, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/missionresponse/special/MissionResponse3540.hpp>
#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Response to a mission definition.
//!
//! Special data type: \ref microvision::common::sdk::MissionResponse3540
//------------------------------------------------------------------------------
class MissionResponse final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    MissionResponse();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~MissionResponse() = default;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.missionresponse"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //!\brief Set the timestamp of the MissionResponse.
    //!\param[in] timestamp  New timestamp of the MissionResponse.
    //----------------------------------------
    void setTimestamp(const NtpTime& timestamp) { m_delegate.setTimestamp(timestamp); }

    //========================================
    //!\brief Set the response of the MissionResponse.
    //!\param[in] response  New response of the MissionResponse.
    //----------------------------------------
    void setResponse(const MissionResponse3540::Response response) { m_delegate.setResponse(response); }

    //========================================
    //!\brief Set the ID of the MissionResponse.
    //!\param[in] missionId  New ID of the MissionResponse.
    //----------------------------------------
    void setMissionId(const uint32_t missionId) { m_delegate.setMissionId(missionId); }

public:
    //========================================
    //!\brief Get the timestamp of the MissionResponse.
    //!\return timestamp of the MissionResponse.
    //----------------------------------------
    const NtpTime& getTimestamp() const { return m_delegate.getTimestamp(); }

    //========================================
    //!\brief Get the response of the MissionResponse.
    //!\return response of the MissionResponse.
    //----------------------------------------
    MissionResponse3540::Response getResponse() const { return m_delegate.getResponse(); }

    //========================================
    //!\brief Get the id of the MissionResponse.
    //!\return missionId of the MissionResponse.
    //----------------------------------------
    uint32_t getMissionId() const { return m_delegate.getMissionId(); }

    //==============================================================================
    //! \brief Returns value from m_reserved array at specified position \a idx.
    //! \param[in] idx  Index specifying position in m_reserved array.
    //! \return Value of m_reserved array at position \a idx. Returns last
    //!         element of array, if \a idx >= \a nbOfReserved.
    //------------------------------------------------------------------------------
    uint32_t getReserved(const uint_fast8_t idx) const { return m_delegate.getReserved(idx); };

protected:
    MissionResponse3540 m_delegate; //< only possible specialization currently
}; // MissionResponse

//==============================================================================

bool operator==(const MissionResponse& lhs, const MissionResponse& rhs);
bool operator!=(const MissionResponse& lhs, const MissionResponse& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
