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
//! \date Dec 3rd, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

// all exporter includes required to register
// included here in registration order

#include <microvision/common/sdk/datablocks/canmessage/CanMessage1002Exporter1002.hpp>
#include <microvision/common/sdk/datablocks/canmessage/CanMessageExporter1002.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayList6970Exporter6970.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayList6972Exporter6972.hpp>
#include <microvision/common/sdk/datablocks/carriageway/CarriageWayListExporter6972.hpp>

#include <microvision/common/sdk/datablocks/commands/Command2010Exporter2010.hpp>

#include <microvision/common/sdk/datablocks/contentseparator/special/ContentSeparator7100Exporter7100.hpp>
#include <microvision/common/sdk/datablocks/contentseparator/ContentSeparatorExporter7100.hpp>

#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainerExporter.hpp>

#include <microvision/common/sdk/datablocks/destination/special/Destination3520Exporter3520.hpp>
#include <microvision/common/sdk/datablocks/destination/special/Destination3521Exporter3521.hpp>
#include <microvision/common/sdk/datablocks/destination/DestinationExporter3520.hpp>
#include <microvision/common/sdk/datablocks/destination/DestinationExporter3521.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6301Exporter6301.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6303Exporter6303.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6320Exporter6320.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/DeviceStatusExporter6301.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/DeviceStatusExporter6303.hpp>

#include <microvision/common/sdk/datablocks/eventtag/special/EventTag7000Exporter7000.hpp>
#include <microvision/common/sdk/datablocks/eventtag/EventTagExporter7000.hpp>

#include <microvision/common/sdk/datablocks/frameendseparator/FrameEndSeparator1100Exporter1100.hpp>

#include <microvision/common/sdk/datablocks/frameindex/special/FrameIndex6130Exporter6130.hpp>

#include <microvision/common/sdk/datablocks/frameindex/FrameIndexExporter6130.hpp>

#include <microvision/common/sdk/datablocks/geoframeindex/special/GeoFrameIndex6140Exporter6140.hpp>
#include <microvision/common/sdk/datablocks/geoframe/special/GeoFrameStart6150Exporter6150.hpp>

#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImu9001Exporter9001.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImu9004Exporter9004.hpp>

#include <microvision/common/sdk/datablocks/gpsimu/GpsImuExporter9001.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/GpsImuExporter9004.hpp>

#include <microvision/common/sdk/datablocks/eventmarker/special/EventMarker7001Exporter7001.hpp>

#include <microvision/common/sdk/datablocks/eventmarker/EventMarkerExporter7001.hpp>

#include <microvision/common/sdk/datablocks/idctrailer/special/IdcTrailer6120Exporter6120.hpp>

#include <microvision/common/sdk/datablocks/idctrailer/IdcTrailerExporter6120.hpp>

#include <microvision/common/sdk/datablocks/idsequence/IdSequence3500Exporter3500.hpp>

#include <microvision/common/sdk/datablocks/idsequence/IdSequenceExporter3500.hpp>

#include <microvision/common/sdk/datablocks/image/special/Image2403Exporter2403.hpp>
#include <microvision/common/sdk/datablocks/image/ImageExporter2403.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2404Exporter2404.hpp>
#include <microvision/common/sdk/datablocks/image/ImageExporter2404.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2405Exporter2405.hpp>
#include <microvision/common/sdk/datablocks/image/ImageExporter2405.hpp>

#include <microvision/common/sdk/datablocks/lanemarking/special/LaneMarkingList6901Exporter6901.hpp>
#include <microvision/common/sdk/datablocks/lanemarking/LaneMarkingListExporter6901.hpp>

#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundaryList6902Exporter6902.hpp>

#include <microvision/common/sdk/datablocks/ldmiAggregated/special/LdmiAggregatedFrame2355Exporter2355.hpp>

#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2352Exporter2352.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2353Exporter2353.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2354Exporter2354.hpp>

#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageDebug6430Exporter6430.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageError6400Exporter6400.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessageExporter6400.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessageExporter6410.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessageExporter6420.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessageExporter6430.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageNote6420Exporter6420.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageWarning6410Exporter6410.hpp>

#include <microvision/common/sdk/datablocks/logpolygonlist2d/special/LogPolygonList2dFloat6817Exporter6817.hpp>
#include <microvision/common/sdk/datablocks/logpolygonlist2d/LogPolygonList2dExporter6817.hpp>

#include <microvision/common/sdk/datablocks/marker/special/MarkerList6820Exporter6820.hpp>

#include <microvision/common/sdk/datablocks/mavinRaw/special/MavinRawFrame2360Exporter2360.hpp>

#include <microvision/common/sdk/datablocks/measurementlist/special/MeasurementList2821Exporter2821.hpp>
#include <microvision/common/sdk/datablocks/measurementlist/MeasurementListExporter2821.hpp>

#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationList7110Exporter7110.hpp>
#include <microvision/common/sdk/datablocks/metainformationlist/MetaInformationListExporter7110.hpp>

#include <microvision/common/sdk/datablocks/missionhandling/MissionHandlingStatus3530Exporter3530.hpp>

#include <microvision/common/sdk/datablocks/missionresponse/special/MissionResponse3540Exporter3540.hpp>
#include <microvision/common/sdk/datablocks/missionresponse/MissionResponseExporter3540.hpp>

#include <microvision/common/sdk/datablocks/notification/special/Notification2030Exporter2030.hpp>
#include <microvision/common/sdk/datablocks/notification/NotificationExporter2030.hpp>

#include <microvision/common/sdk/datablocks/objectassociationlist/special/ObjectAssociationList4001Exporter4001.hpp>
#include <microvision/common/sdk/datablocks/objectassociationlist/ObjectAssociationListExporter4001.hpp>

#include <microvision/common/sdk/datablocks/objectlabellist/special/ObjectLabelList6503Exporter6503.hpp>
#include <microvision/common/sdk/datablocks/objectlabellist/ObjectLabelListExporter6503.hpp>

#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2221Exporter2221.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2225Exporter2225.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2270Exporter2270.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2271Exporter2271.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2280Exporter2280.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2281Exporter2281.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2290Exporter2290.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2291Exporter2291.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListExporter2271.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListExporter2281.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListExporter2291.hpp>

#include <microvision/common/sdk/datablocks/odometry/special/Odometry9002Exporter9002.hpp>
#include <microvision/common/sdk/datablocks/odometry/special/Odometry9003Exporter9003.hpp>
#include <microvision/common/sdk/datablocks/odometry/OdometryExporter9002.hpp>
#include <microvision/common/sdk/datablocks/odometry/OdometryExporter9003.hpp>

#include <microvision/common/sdk/datablocks/ogpsimumessage/special/OGpsImuMessage2610Exporter2610.hpp>
#include <microvision/common/sdk/datablocks/ogpsimumessage/OGpsImuMessageExporter2610.hpp>

#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7500Exporter7500.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7510Exporter7510.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7511Exporter7510.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7511Exporter7511.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/PointCloudExporter7510.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/PointCloudExporter7511.hpp>

#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84_2604Exporter2604.hpp>
#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84Sequence3510Exporter3510.hpp>
#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84SequenceExporter2604.hpp>
#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84SequenceExporter3510.hpp>

#include <microvision/common/sdk/datablocks/scan/special/Scan2202Exporter2202.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2205Exporter2205.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2208Exporter2208.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2209Exporter2209.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2310Exporter2310.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2321Exporter2321.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340Exporter2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2341/Scan2341Exporter2341.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342Exporter2342.hpp>
#include <microvision/common/sdk/datablocks/scan/ScanExporter2209.hpp>

#include <microvision/common/sdk/datablocks/stateofoperation/special/StateOfOperation9110Exporter9110.hpp>
#include <microvision/common/sdk/datablocks/stateofoperation/special/StateOfOperation9111Exporter9111.hpp>
#include <microvision/common/sdk/datablocks/stateofoperation/StateOfOperationExporter9111.hpp>

#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringCanStatus6700Exporter6700.hpp>
#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringDeviceStatus6701Exporter6701.hpp>
#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringSystemStatus6705Exporter6705.hpp>

#include <microvision/common/sdk/datablocks/timerecord/special/TimeRecord9000Exporter9000.hpp>
#include <microvision/common/sdk/datablocks/timerecord/TimeRecordExporter9000.hpp>

#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationsList9010Exporter9010.hpp>
#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationsList9011Exporter9011.hpp>
#include <microvision/common/sdk/datablocks/timerelation/TimeRelationsListExporter9011.hpp>

#include <microvision/common/sdk/datablocks/trafficlight/special/TrafficLightStateList3600Exporter3600.hpp>
#include <microvision/common/sdk/datablocks/trafficlight/TrafficLightStateListExporter3600.hpp>

#include <microvision/common/sdk/datablocks/eventtag/special/UserEventTag7010Exporter7010.hpp>

#include <microvision/common/sdk/datablocks/vehiclecontrol/special/VehicleControl9100Exporter9100.hpp>
#include <microvision/common/sdk/datablocks/vehiclecontrol/VehicleControlExporter9100.hpp>

#include <microvision/common/sdk/datablocks/vehiclerequest/special/VehicleRequest9105Exporter9105.hpp>
#include <microvision/common/sdk/datablocks/vehiclerequest/VehicleRequestExporter9105.hpp>

#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2805Exporter2805.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2806Exporter2806.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2807Exporter2807.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2808Exporter2808.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2809Exporter2809.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleStateExporter2805.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleStateExporter2808.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleStateExporter2809.hpp>

#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/ZoneOccupationListA000ExporterA000.hpp>
