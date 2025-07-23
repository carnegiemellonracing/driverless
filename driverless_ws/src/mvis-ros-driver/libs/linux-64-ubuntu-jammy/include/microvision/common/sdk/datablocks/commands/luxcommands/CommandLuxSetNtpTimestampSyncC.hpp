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
//! \date Feb 19, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/commands/luxcommands/LuxCommandC.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandLuxSetNtpTimestampSyncC final : public LuxCommandC<CommandId::Id::CmdLuxSetNtpTimestampSync>
{
    template<class ContainerType, DataTypeId::DataType id, class SpecialCommand>
    friend class SpecialImporter;
    template<class SpecialCommand>
    friend class SpecialExporter;

public:
    //========================================
    //! \brief Length of the SetFilter command.
    //----------------------------------------
    constexpr static const KeyType key{CommandId::Id::CmdLuxSetNtpTimestampSync};
    constexpr static const char* const containerType{"sdk.specialcontainer.command2010.commandluxsetntptimestampsync"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    CommandLuxSetNtpTimestampSyncC() : CommandLuxSetNtpTimestampSyncC(NtpTime{0}) {}

    CommandLuxSetNtpTimestampSyncC(const NtpTime timestamp)
      : LuxCommandC<CommandId::Id::CmdLuxSetNtpTimestampSync>(), m_timestamp(timestamp)
    {}

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint16_t getReserved0() const { return m_reserved0; }
    uint16_t getReserved1() const { return m_reserved1; }
    NtpTime getTimestamp() const { return m_timestamp; }

public:
    void setTimestamp(const NtpTime timestamp) { m_timestamp = timestamp; }

protected:
    uint16_t m_reserved0{0x0000U};
    uint16_t m_reserved1{0x0000U};
    NtpTime m_timestamp;
}; // CommandLuxSetNtpTimestampSyncC

//==============================================================================
//==============================================================================
//==============================================================================

inline bool operator==(const CommandLuxSetNtpTimestampSyncC& lhs, const CommandLuxSetNtpTimestampSyncC& rhs)
{
    return static_cast<const CommandCBase&>(lhs) == static_cast<const CommandCBase&>(rhs)
           && lhs.getReserved0() == rhs.getReserved0() && lhs.getReserved1() == rhs.getReserved1()
           && lhs.getReserved0() == rhs.getTimestamp();
}

//==============================================================================

inline bool operator!=(const CommandLuxSetNtpTimestampSyncC& lhs, const CommandLuxSetNtpTimestampSyncC& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
