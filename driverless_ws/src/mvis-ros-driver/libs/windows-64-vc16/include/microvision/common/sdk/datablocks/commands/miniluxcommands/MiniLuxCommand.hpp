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

#include <microvision/common/sdk/datablocks/commands/Command.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class MiniLuxCommandReplyBase : public CommandReplyBase
{
public:
    MiniLuxCommandReplyBase(const CommandId commandId) : CommandReplyBase(commandId) {}
}; // MiniLuxCommandReplyBase

//==============================================================================

template<CommandId::Id cId>
class MiniLuxCommandReply : public MiniLuxCommandReplyBase
{
public:
    MiniLuxCommandReply() : MiniLuxCommandReplyBase(cId) {}
}; // MiniLuxCommandReply<cId>

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
