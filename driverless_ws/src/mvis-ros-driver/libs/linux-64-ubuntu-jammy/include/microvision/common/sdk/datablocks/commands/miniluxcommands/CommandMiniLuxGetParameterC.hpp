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
//! \date Feb 14, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/commands/miniluxcommands/MiniLuxCommandC.hpp>
#include <microvision/common/sdk/misc/ParameterIndex.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandMiniLuxGetParameterC final : public MiniLuxCommandC<CommandId::Id::CmdLuxGetParameter>
{
    template<class ContainerType, DataTypeId::DataType id, class SpecialCommand>
    friend class SpecialImporter;
    template<class SpecialCommand>
    friend class SpecialExporter;

public:
    //========================================
    //! \brief Length of the SetFilter command.
    //----------------------------------------
    constexpr static const KeyType key{CommandId::Id::CmdLuxGetParameter};
    constexpr static const char* const containerType{"sdk.specialcontainer.command.commandminiluxgetparameter"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    CommandMiniLuxGetParameterC() : CommandMiniLuxGetParameterC(ParameterIndex{0x0000U}) {}
    explicit CommandMiniLuxGetParameterC(const ParameterIndex parameterIndex)
      : MiniLuxCommandC<CommandId::Id::CmdLuxGetParameter>(), m_parameterIndex(parameterIndex)
    {}

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint16_t getReserved() const { return m_reserved; }
    ParameterIndex getParameterIndex() const { return m_parameterIndex; }

public:
    void setParameterIndex(const ParameterIndex& idx) { m_parameterIndex = idx; }

protected:
    uint16_t m_reserved{0x0000U};
    ParameterIndex m_parameterIndex;
}; // CommandMiniLuxGetParameterC

//==============================================================================
//==============================================================================
//==============================================================================

inline bool operator==(const CommandMiniLuxGetParameterC& lhs, const CommandMiniLuxGetParameterC& rhs)
{
    return static_cast<const CommandCBase&>(lhs) == static_cast<const CommandCBase&>(rhs)
           && lhs.getReserved() == rhs.getReserved() && lhs.getParameterIndex() == rhs.getParameterIndex();
}

//==============================================================================

inline bool operator!=(const CommandMiniLuxGetParameterC& lhs, const CommandMiniLuxGetParameterC& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
