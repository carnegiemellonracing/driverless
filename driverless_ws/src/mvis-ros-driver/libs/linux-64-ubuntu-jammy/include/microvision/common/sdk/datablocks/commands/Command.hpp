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
//! \date Sep 5, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/CommandId.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================
//! \class CommandReply
//! \brief Abstract base class for all CommandReply classes.
//! \date Sep 5, 2013
//!
//------------------------------------------------------------------------------
class CommandReplyBase
{
public:
    static const uint16_t errorMask = 0x8000;

public:
    //========================================
    //! \brief Create a Command object.
    //! \param[in] commandId      Id of the command.
    //----------------------------------------
    CommandReplyBase(const CommandId commandId) : m_commandId(commandId) {}

    virtual ~CommandReplyBase() {}

public:
    //========================================
    //! \brief Get the id of this Command.
    //! \return the id of this Command.
    //----------------------------------------
    CommandId getCommandId() const { return m_commandId; }

    virtual bool deserializeFromStream(std::istream& is, const IdcDataHeader& dh) = 0;

public:
    bool isErrorReply() const { return (uint16_t(getCommandId()) & errorMask) != 0; }
    void setErrorReply() { m_commandId = CommandId(uint16_t(m_commandId) | errorMask); }
    void unsetErrorReply() { m_commandId = CommandId(uint16_t(uint16_t(m_commandId) & uint16_t(~errorMask))); }

protected:
    //========================================
    //! \brief The id of this Command.
    //----------------------------------------
    CommandId m_commandId;
}; // CommandReplyBase

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
