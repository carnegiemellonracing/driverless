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

#include <microvision/common/sdk/datablocks/commands/Command2010.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Base class for LUX commands
//!\date Feb 14, 2018
//------------------------------------------------------------------------------
class LuxCommandCBase : public CommandCBase
{
public:
    LuxCommandCBase(const CommandId commandId) : CommandCBase(commandId) {}
}; // LuxCommandBase

//==============================================================================
//!\brief Template base class for LUX commands with given command id.
//!\date Feb 14, 2018
//!\tparam cId  Command id of the LUX command.
//------------------------------------------------------------------------------
template<CommandId::Id cId>
class LuxCommandC : public LuxCommandCBase
{
public:
    LuxCommandC() : LuxCommandCBase(cId) {}
}; // LuxCommand<cId>

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
