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

#include <microvision/common/sdk/datablocks/commands/miniluxcommands/MiniLuxCommandC.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandMiniLuxStartMeasureC final : public MiniLuxCommandC<CommandId::Id::CmdLuxStartMeasure>
{
    template<class ContainerType, DataTypeId::DataType id, class SpecialCommand>
    friend class SpecialImporter;
    template<class SpecialCommand>
    friend class EmptyCommandExporter;

public:
    //========================================
    //! \brief Length of the SetFilter command.
    //----------------------------------------
    constexpr static const KeyType key{CommandId::Id::CmdLuxStartMeasure};
    constexpr static const char* const containerType{"sdk.specialcontainer.command2010.commandminiluxstartmeasure"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    CommandMiniLuxStartMeasureC() : MiniLuxCommandC<CommandId::Id::CmdLuxStartMeasure>() {}

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint16_t getReserved() const { return m_reserved; }

protected:
    uint16_t m_reserved{0x0000U};
}; // CommandMiniLuxStartMeasureC

//==============================================================================
//==============================================================================
//==============================================================================

inline bool operator==(const CommandMiniLuxStartMeasureC& lhs, const CommandMiniLuxStartMeasureC& rhs)
{
    return static_cast<const CommandCBase&>(lhs) == static_cast<const CommandCBase&>(rhs)
           && lhs.getReserved() == rhs.getReserved();
}

//==============================================================================

inline bool operator!=(const CommandMiniLuxStartMeasureC& lhs, const CommandMiniLuxStartMeasureC& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
