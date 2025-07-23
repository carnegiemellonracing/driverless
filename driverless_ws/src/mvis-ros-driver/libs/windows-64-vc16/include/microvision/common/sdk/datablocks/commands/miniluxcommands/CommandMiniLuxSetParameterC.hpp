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
#include <microvision/common/sdk/misc/ParameterData.hpp>
#include <microvision/common/sdk/misc/ParameterIndex.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandMiniLuxSetParameterC final : public MiniLuxCommandC<CommandId::Id::CmdLuxSetParameter>
{
    template<class ContainerType, DataTypeId::DataType id, class SpecialCommand>
    friend class SpecialImporter;
    template<class SpecialCommand>
    friend class SpecialExporter;

public:
    //========================================
    //! \brief Length of the SetFilter command.
    //----------------------------------------
    constexpr static const KeyType key{CommandId::Id::CmdLuxSetParameter};
    constexpr static const char* const containerType{"sdk.specialcontainer.command2010.commandminiluxsetparameter"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    CommandMiniLuxSetParameterC() : CommandMiniLuxSetParameterC(ParameterIndex{0x0000U}, ParameterData{0xFFFFFFFFU}) {}
    CommandMiniLuxSetParameterC(const ParameterIndex parameterIndex, const ParameterData parameterData)
      : MiniLuxCommandC<CommandId::Id::CmdLuxSetParameter>(),
        m_parameterIndex(parameterIndex),
        m_parameterData(parameterData)
    {}

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint16_t getReserved() const { return m_reserved; }
    ParameterIndex getParameterIndex() const { return m_parameterIndex; }
    ParameterData getParameterData() const { return m_parameterData; }

public:
    void setParameterIndex(const ParameterIndex idx) { m_parameterIndex = idx; }
    void setParameterData(const ParameterData data) { m_parameterData = data; }

protected:
    uint16_t m_reserved{0x0000U};
    ParameterIndex m_parameterIndex;
    ParameterData m_parameterData;
}; // CommandMiniLuxSetParameterC

//==============================================================================
//==============================================================================
//==============================================================================

inline bool operator==(const CommandMiniLuxSetParameterC& lhs, const CommandMiniLuxSetParameterC& rhs)
{
    return static_cast<const CommandCBase&>(lhs) == static_cast<const CommandCBase&>(rhs)
           && lhs.getReserved() == rhs.getReserved() && lhs.getParameterIndex() == rhs.getParameterIndex()
           && lhs.getParameterData() == rhs.getParameterData();
}

//==============================================================================

inline bool operator!=(const CommandMiniLuxSetParameterC& lhs, const CommandMiniLuxSetParameterC& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
