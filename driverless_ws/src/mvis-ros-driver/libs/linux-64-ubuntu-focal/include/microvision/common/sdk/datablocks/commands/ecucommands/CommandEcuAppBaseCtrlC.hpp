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

#include <microvision/common/sdk/datablocks/commands/ecucommands/EcuCommandC.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandEcuAppBaseCtrlC final : public EcuCommandC<CommandId::Id::CmdManagerAppBaseCtrl>
{
    template<class ContainerType, DataTypeId::DataType id, class SpecialCommand>
    friend class SpecialImporter;
    template<class SpecialCommand>
    friend class SpecialExporter;

public:
    enum class AppBaseCtrlId : uint16_t
    {
        Invalid        = 0x0000,
        StartRecording = 0x0001,
        StopRecording  = 0x0002
    }; // AppBaseCtrlId

public:
    //========================================
    //! \brief Length of the SetFilter command.
    //----------------------------------------
    constexpr static const KeyType key{CommandId::Id::CmdManagerAppBaseCtrl};
    constexpr static const char* const containerType{"sdk.specialcontainer.command.commandecuappbasectrl"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    CommandEcuAppBaseCtrlC() : CommandEcuAppBaseCtrlC(AppBaseCtrlId::Invalid, std::string()) {}
    CommandEcuAppBaseCtrlC(const AppBaseCtrlId ctrlId) : CommandEcuAppBaseCtrlC(ctrlId, std::string()) {}
    CommandEcuAppBaseCtrlC(const AppBaseCtrlId ctrlId, const std::string& data)
      : EcuCommandC<CommandId::Id::CmdManagerAppBaseCtrl>(), m_ctrlId(ctrlId), m_data(data)
    {}

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    AppBaseCtrlId getCtrlId() const { return this->m_ctrlId; }
    const std::string& getData() const { return this->m_data; }

public:
    void setContent(const AppBaseCtrlId ctrlId, const std::string& data)
    {
        m_ctrlId = ctrlId;
        m_data   = data;
    }

protected:
    AppBaseCtrlId m_ctrlId;
    std::string m_data;
}; // CommandManagerAppBaseCtrl

//==============================================================================
//==============================================================================
//==============================================================================

inline bool operator==(const CommandEcuAppBaseCtrlC& lhs, const CommandEcuAppBaseCtrlC& rhs)
{
    return static_cast<const CommandCBase&>(lhs) == static_cast<const CommandCBase&>(rhs)
           && lhs.getCtrlId() == rhs.getCtrlId() && lhs.getData() == rhs.getData();
}

//==============================================================================

inline bool operator!=(const CommandEcuAppBaseCtrlC& lhs, const CommandEcuAppBaseCtrlC& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
