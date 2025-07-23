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
//!\brief Base class for ECU commands
//!\date Feb 14, 2018
//------------------------------------------------------------------------------
class EcuCommandCBase : public CommandCBase
{
public:
    EcuCommandCBase(const CommandId commandId) : CommandCBase(commandId) {}
}; // EcuCommandCBase

//==============================================================================
//!\brief Template base class for ECU commands with given command id.
//!\date Feb 14, 2018
//!\tparam cId  Command id of the ECU command.
//------------------------------------------------------------------------------
template<CommandId::Id cId>
class EcuCommandC : public EcuCommandCBase
{
public:
    EcuCommandC() : EcuCommandCBase(cId) {}
}; // EcuCommandC<cId>

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
