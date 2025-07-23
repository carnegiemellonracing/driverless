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
//! \date Feb 16, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/commands/ecucommands/EcuCommandC.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandEcuSetFilterC final : public EcuCommandC<CommandId::Id::CmdManagerSetFilter>
{
    template<class ContainerType, DataTypeId::DataType id, class SpecialCommand>
    friend class SpecialImporter;
    template<class SpecialCommand>
    friend class SpecialExporter;

public:
    using Range       = std::pair<DataTypeId, DataTypeId>;
    using RangeVector = std::vector<Range>;

public:
    constexpr static const KeyType key{CommandId::Id::CmdManagerSetFilter};
    constexpr static const char* const containerType{"sdk.specialcontainer.command.commandecusetfilter"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    CommandEcuSetFilterC() : CommandEcuSetFilterC(DataTypeId::DataType_Unknown, DataTypeId::DataType_LastId) {}

    CommandEcuSetFilterC(RangeVector&& ranges) : EcuCommandC<CommandId::Id::CmdManagerSetFilter>(), m_ranges(ranges) {}

    CommandEcuSetFilterC(const RangeVector& ranges)
      : EcuCommandC<CommandId::Id::CmdManagerSetFilter>(), m_ranges(ranges)
    {}

    //========================================
    //!\brief DataType range of the output filter.
    //!
    //!Datatypes with id between \a rangeStart and
    //!\a rangeEnd shall pass the output filter.
    //!\param[in] rangeStart  Lowest DataType it
    //!                       to be passed the filter.
    //!\param[in] rangeEnd    Highest DataType it
    //!                       to be passed the filter.
    //----------------------------------------
    CommandEcuSetFilterC(const DataTypeId rangeStart, const DataTypeId rangeEnd)
      : CommandEcuSetFilterC{RangeVector{std::make_pair(rangeStart, rangeEnd)}}
    {}

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //!\brief Get the version of this Command.
    //!\return The version of this command.
    //----------------------------------------
    uint16_t getNbOfRanges() const { return static_cast<uint16_t>(m_ranges.size()); }

    const RangeVector& getRanges() const { return m_ranges; }

public:
    void setRanges(const RangeVector& ranges) { m_ranges = ranges; }

protected:
    //========================================
    //!\brief The version of this Command.
    //----------------------------------------
    RangeVector m_ranges;
}; // CommandEcuSetFilterC

//==============================================================================
//==============================================================================
//==============================================================================

inline bool operator==(const CommandEcuSetFilterC& lhs, const CommandEcuSetFilterC& rhs)
{
    return static_cast<const CommandCBase&>(lhs) == static_cast<const CommandCBase&>(rhs)
           && lhs.getRanges() == rhs.getRanges();
}

//==============================================================================

inline bool operator!=(const CommandEcuSetFilterC& lhs, const CommandEcuSetFilterC& rhs)
{
    return static_cast<const CommandCBase&>(lhs) != static_cast<const CommandCBase&>(rhs)
           || lhs.getRanges() != rhs.getRanges();
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
