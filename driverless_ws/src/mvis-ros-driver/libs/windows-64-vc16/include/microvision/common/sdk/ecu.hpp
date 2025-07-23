//==============================================================================
//! \file
//! \brief Include file for using MvisEcu.
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

#include <microvision/common/sdk/devices/MvisEcuDevice.hpp>

#include <microvision/common/sdk/datablocks/canmessage/CanMessage1002.hpp>
#include <microvision/common/sdk/datablocks/frameendseparator/FrameEndSeparator1100.hpp>

#include <microvision/common/sdk/datablocks/scan/Scan.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2202.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2205.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2208.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2209.hpp>

#include <microvision/common/sdk/datablocks/objectlist/ObjectList.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2221.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2225.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2270.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2271.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2280.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2281.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2291.hpp>

#include <microvision/common/sdk/datablocks/image/Image.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2403.hpp>

#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84_2604.hpp>

#include <microvision/common/sdk/datablocks/ogpsimumessage/OGpsImuMessage.hpp>
#include <microvision/common/sdk/datablocks/ogpsimumessage/special/OGpsImuMessage2610.hpp>

#include <microvision/common/sdk/datablocks/vehiclestate/VehicleState.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2805.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2806.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2807.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2808.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2809.hpp>

#include <microvision/common/sdk/datablocks/measurementlist/special/MeasurementList2821.hpp>

#include <microvision/common/sdk/datablocks/objectassociationlist/ObjectAssociationList.hpp>
#include <microvision/common/sdk/datablocks/objectassociationlist/special/ObjectAssociationList4001.hpp>

#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringCanStatus6700.hpp>
#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringDeviceStatus6701.hpp>
#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringSystemStatus6705.hpp>

#include <microvision/common/sdk/datablocks/contentseparator/ContentSeparator.hpp>
#include <microvision/common/sdk/datablocks/contentseparator/special/ContentSeparator7100.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/DeviceStatus.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6301.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6303.hpp>

#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageDebug6430.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageError6400.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageNote6420.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageWarning6410.hpp>

#include <microvision/common/sdk/datablocks/objectlabellist/ObjectLabelList.hpp>
#include <microvision/common/sdk/datablocks/objectlabellist/special/ObjectLabelList6503.hpp>

#include <microvision/common/sdk/datablocks/logpolygonlist2d/LogPolygonList2d.hpp>
#include <microvision/common/sdk/datablocks/logpolygonlist2d/special/LogPolygonList2dFloat6817.hpp>

#include <microvision/common/sdk/datablocks/metainformationlist/MetaInformationList.hpp>
#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationList7110.hpp>

#include <microvision/common/sdk/datablocks/pointcloud/PointCloud.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7500.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7510.hpp>

#include <microvision/common/sdk/datablocks/timerecord/special/TimeRecord9000.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/GpsImu.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImu9001.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImu9004.hpp>
#include <microvision/common/sdk/datablocks/odometry/Odometry.hpp>
#include <microvision/common/sdk/datablocks/odometry/special/Odometry9002.hpp>

#include <microvision/common/sdk/extension/DeviceFactory.hpp>
#include <microvision/common/sdk/extension/ConfigurationFactory.hpp>

#include <microvision/common/sdk/listener/DataStreamer.hpp>
#include <microvision/common/sdk/listener/DataContainerListener.hpp>

#include <microvision/common/sdk/MicroVisionSdk.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
