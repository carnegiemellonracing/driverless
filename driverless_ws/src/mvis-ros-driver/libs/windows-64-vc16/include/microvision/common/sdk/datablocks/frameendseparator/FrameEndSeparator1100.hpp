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
//! \date Jan 11, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Idc frame end separator
//!
//! FrameEndSeparator is placed at the end of one frame (e.g. frame like ScanPtList
//! with vehicleData, DeviceStatus and ObjectData).
//------------------------------------------------------------------------------
class FrameEndSeparator1100 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static const uint8_t nbOfReserved = 6;

public:
    using ReservedArray = std::array<uint16_t, nbOfReserved>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.frameendseparator1100"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    FrameEndSeparator1100();
    virtual ~FrameEndSeparator1100();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint32_t getFrameId() const { return m_frameId; }
    uint32_t getSizeOfThisFrame() const { return m_sizeOfThisFrame; }
    uint32_t getSizeOfNextFrame() const { return m_sizeOfNextFrame; }
    NtpTime getCreationTime() const { return m_creationTime; }

    uint16_t getReserved(const uint8_t idx) const { return m_reserved.at(idx); }
    const ReservedArray& getReserved() const { return m_reserved; }

public:
    void setFrameId(const uint32_t newFrameId) { m_frameId = newFrameId; }
    void setSizeOfThisFrame(const uint32_t newSizeOfThisFrame) { m_sizeOfThisFrame = newSizeOfThisFrame; }
    void setSizeOfNextFrame(const uint32_t newSizeOfNextFrame) { m_sizeOfNextFrame = newSizeOfNextFrame; }
    void setCreationTime(const NtpTime newCreationTime) { m_creationTime = newCreationTime; }

protected:
    uint32_t m_frameId{0};
    uint32_t m_sizeOfThisFrame{0};
    uint32_t m_sizeOfNextFrame{0};
    NtpTime m_creationTime{};
    ReservedArray m_reserved{{0x0, 0x0, 0x0, 0x0, 0x0, 0x0}};
}; // FrameEndSeparator1100

//==============================================================================

bool operator==(const FrameEndSeparator1100& lhs, const FrameEndSeparator1100& rhs);
bool operator!=(const FrameEndSeparator1100& lhs, const FrameEndSeparator1100& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
