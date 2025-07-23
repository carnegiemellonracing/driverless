//==============================================================================
//! \file
//! \brief Include file for using MvisTrackingBox.
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Sep 30, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/MvisTrackingBoxDevice.hpp>
#include <microvision/common/sdk/config/devices/MvisEcuConfiguration.hpp>

#include <microvision/common/sdk/extension/DeviceFactory.hpp>
#include <microvision/common/sdk/extension/ConfigurationFactory.hpp>

#include <microvision/common/sdk/datablocks/scan/Scan.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2202.hpp>

#include <microvision/common/sdk/datablocks/objectlist/ObjectList.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2221.hpp>

#include <microvision/common/sdk/datablocks/vehiclestate/VehicleState.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2805.hpp>

#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageDebug6430.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageError6400.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageNote6420.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageWarning6410.hpp>

#include <microvision/common/sdk/listener/DataStreamer.hpp>

#include <microvision/common/sdk/MicroVisionSdk.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
