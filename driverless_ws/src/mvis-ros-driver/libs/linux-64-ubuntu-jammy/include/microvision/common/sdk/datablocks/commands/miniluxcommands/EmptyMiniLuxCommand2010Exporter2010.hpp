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
//! \date Mar 1, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/commands/EmptyCommandExporter.hpp>
#include <microvision/common/sdk/datablocks/commands/miniluxcommands/CommandMiniLuxGetStatusC.hpp>
#include <microvision/common/sdk/datablocks/commands/miniluxcommands/CommandMiniLuxSaveConfigC.hpp>
#include <microvision/common/sdk/datablocks/commands/miniluxcommands/CommandMiniLuxResetC.hpp>
#include <microvision/common/sdk/datablocks/commands/miniluxcommands/CommandMiniLuxResetToDefaultParametersC.hpp>
#include <microvision/common/sdk/datablocks/commands/miniluxcommands/CommandMiniLuxStartMeasureC.hpp>
#include <microvision/common/sdk/datablocks/commands/miniluxcommands/CommandMiniLuxStopMeasureC.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class SpecialExporter<CommandMiniLuxGetStatusC> : public EmptyCommandExporter<CommandMiniLuxGetStatusC>
{
};
template<>
class SpecialExporter<CommandMiniLuxSaveConfigC> : public EmptyCommandExporter<CommandMiniLuxSaveConfigC>
{
};
template<>
class SpecialExporter<CommandMiniLuxResetC> : public EmptyCommandExporter<CommandMiniLuxResetC>
{
};
template<>
class SpecialExporter<CommandMiniLuxResetToDefaultParametersC>
  : public EmptyCommandExporter<CommandMiniLuxResetToDefaultParametersC>
{
};
template<>
class SpecialExporter<CommandMiniLuxStartMeasureC> : public EmptyCommandExporter<CommandMiniLuxStartMeasureC>
{
};
template<>
class SpecialExporter<CommandMiniLuxStopMeasureC> : public EmptyCommandExporter<CommandMiniLuxStopMeasureC>
{
};

//==============================================================================

using CommandMiniLuxGetStatus2010Exporter2010                = SpecialExporter<CommandMiniLuxGetStatusC>;
using CommandMiniLuxSaveConfig2010Exporter2010               = SpecialExporter<CommandMiniLuxSaveConfigC>;
using CommandMiniLuxReset2010Exporter2010                    = SpecialExporter<CommandMiniLuxResetC>;
using CommandMiniLuxResetToDefaultParameters2010Exporter2010 = SpecialExporter<CommandMiniLuxResetToDefaultParametersC>;
using CommandMiniLuxStartMeasure2010Exporter2010             = SpecialExporter<CommandMiniLuxStartMeasureC>;
using CommandMiniLuxStopMeasure2010Exporter2010              = SpecialExporter<CommandMiniLuxStopMeasureC>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
