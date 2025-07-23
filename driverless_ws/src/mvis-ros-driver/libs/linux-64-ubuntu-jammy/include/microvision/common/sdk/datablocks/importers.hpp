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
//! \date Sep 27, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

// all importer includes required to register
// included here in registration order

#include <microvision/common/sdk/datablocks/canmessage/CanMessage1002Importer1002.hpp>
#include <microvision/common/sdk/datablocks/canmessage/CanMessageImporter1002.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayList6970Importer6970.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayList6972Importer6972.hpp>
#include <microvision/common/sdk/datablocks/carriageway/CarriageWayListImporter6970.hpp>
#include <microvision/common/sdk/datablocks/carriageway/CarriageWayListImporter6972.hpp>

#include <microvision/common/sdk/datablocks/commands/Command2010Importer2010.hpp>

#include <microvision/common/sdk/datablocks/contentseparator/special/ContentSeparator7100Importer7100.hpp>
#include <microvision/common/sdk/datablocks/contentseparator/ContentSeparatorImporter7100.hpp>

#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainerImporter.hpp>

#include <microvision/common/sdk/datablocks/destination/special/Destination3520Importer3520.hpp>
#include <microvision/common/sdk/datablocks/destination/special/Destination3521Importer3521.hpp>
#include <microvision/common/sdk/datablocks/destination/DestinationImporter3520.hpp>
#include <microvision/common/sdk/datablocks/destination/DestinationImporter3521.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6301Importer6301.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6303Importer6303.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6320Importer6320.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/DeviceStatusImporter6301.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/DeviceStatusImporter6303.hpp>

#include <microvision/common/sdk/datablocks/eventtag/special/EventTag7000Importer7000.hpp>
#include <microvision/common/sdk/datablocks/eventtag/EventTagImporter7000.hpp>

#include <microvision/common/sdk/datablocks/frameendseparator/FrameEndSeparator1100Importer1100.hpp>

#include <microvision/common/sdk/datablocks/frameindex/special/FrameIndex6130Importer6130.hpp>

#include <microvision/common/sdk/datablocks/frameindex/FrameIndexImporter6130.hpp>

#include <microvision/common/sdk/datablocks/geoframeindex/special/GeoFrameIndex6140Importer6140.hpp>
#include <microvision/common/sdk/datablocks/geoframe/special/GeoFrameStart6150Importer6150.hpp>

#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImu9001Importer9001.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImu9004Importer9004.hpp>

#include <microvision/common/sdk/datablocks/gpsimu/GpsImuImporter9001.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/GpsImuImporter9004.hpp>

#include <microvision/common/sdk/datablocks/eventmarker/special/EventMarker7001Importer7001.hpp>

#include <microvision/common/sdk/datablocks/eventmarker/EventMarkerImporter7001.hpp>

#include <microvision/common/sdk/datablocks/idctrailer/special/IdcTrailer6120Importer6120.hpp>

#include <microvision/common/sdk/datablocks/idctrailer/IdcTrailerImporter6120.hpp>

#include <microvision/common/sdk/datablocks/idsequence/IdSequence3500Importer3500.hpp>

#include <microvision/common/sdk/datablocks/idsequence/IdSequenceImporter3500.hpp>

#include <microvision/common/sdk/datablocks/image/special/Image2403Importer2403.hpp>
#include <microvision/common/sdk/datablocks/image/ImageImporter2403.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2404Importer2404.hpp>
#include <microvision/common/sdk/datablocks/image/ImageImporter2404.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2405Importer2405.hpp>
#include <microvision/common/sdk/datablocks/image/ImageImporter2405.hpp>

#include <microvision/common/sdk/datablocks/lanemarking/special/LaneMarkingList6901Importer6901.hpp>
#include <microvision/common/sdk/datablocks/lanemarking/LaneMarkingListImporter6901.hpp>

#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundaryList6902Importer6902.hpp>

#include <microvision/common/sdk/datablocks/ldmiAggregated/special/LdmiAggregatedFrame2355Importer2355.hpp>

#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2352Importer2352.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2353Importer2353.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2354Importer2354.hpp>

#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageDebug6430Importer6430.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageError6400Importer6400.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessageImporter6400.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessageImporter6410.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessageImporter6420.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessageImporter6430.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageNote6420Importer6420.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageWarning6410Importer6410.hpp>

#include <microvision/common/sdk/datablocks/logpolygonlist2d/special/LogPolygonList2dFloat6817Importer6817.hpp>
#include <microvision/common/sdk/datablocks/logpolygonlist2d/LogPolygonList2dImporter6817.hpp>

#include <microvision/common/sdk/datablocks/marker/special/MarkerList6820Importer6820.hpp>

#include <microvision/common/sdk/datablocks/mavinRaw/special/MavinRawFrame2360Importer2360.hpp>

#include <microvision/common/sdk/datablocks/measurementlist/special/MeasurementList2821Importer2821.hpp>
#include <microvision/common/sdk/datablocks/measurementlist/MeasurementListImporter2821.hpp>

#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationList7110Importer7110.hpp>
#include <microvision/common/sdk/datablocks/metainformationlist/MetaInformationListImporter7110.hpp>

#include <microvision/common/sdk/datablocks/missionhandling/MissionHandlingStatus3530Importer3530.hpp>

#include <microvision/common/sdk/datablocks/missionresponse/special/MissionResponse3540Importer3540.hpp>
#include <microvision/common/sdk/datablocks/missionresponse/MissionResponseImporter3540.hpp>

#include <microvision/common/sdk/datablocks/notification/special/Notification2030Importer2030.hpp>
#include <microvision/common/sdk/datablocks/notification/NotificationImporter2030.hpp>

#include <microvision/common/sdk/datablocks/objectassociationlist/special/ObjectAssociationList4001Importer4001.hpp>
#include <microvision/common/sdk/datablocks/objectassociationlist/ObjectAssociationListImporter4001.hpp>

#include <microvision/common/sdk/datablocks/objectlabellist/special/ObjectLabelList6503Importer6503.hpp>
#include <microvision/common/sdk/datablocks/objectlabellist/ObjectLabelListImporter6503.hpp>

#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2221Importer2221.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2225Importer2225.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2270Importer2270.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2271Importer2271.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2280Importer2280.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2281Importer2281.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2290Importer2290.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2291Importer2291.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListImporter2221.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListImporter2225.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListImporter2271.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListImporter2280.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListImporter2281.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListImporter2290.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListImporter2291.hpp>

#include <microvision/common/sdk/datablocks/odometry/special/Odometry9002Importer9002.hpp>
#include <microvision/common/sdk/datablocks/odometry/special/Odometry9003Importer9003.hpp>
#include <microvision/common/sdk/datablocks/odometry/OdometryImporter9002.hpp>
#include <microvision/common/sdk/datablocks/odometry/OdometryImporter9003.hpp>

#include <microvision/common/sdk/datablocks/ogpsimumessage/special/OGpsImuMessage2610Importer2610.hpp>
#include <microvision/common/sdk/datablocks/ogpsimumessage/OGpsImuMessageImporter2610.hpp>

#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7500Importer7500.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7510Importer7510.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7511Importer7510.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7511Importer7511.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/PointCloudImporter7510.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/PointCloudImporter7511.hpp>

#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84_2604Importer2604.hpp>
#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84Sequence3510Importer3510.hpp>
#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84SequenceImporter2604.hpp>
#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84SequenceImporter3510.hpp>

#include <microvision/common/sdk/datablocks/scan/special/Scan2202Importer2202.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2205Importer2205.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2208Importer2208.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2208Importer2310.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2209Importer2209.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2209Importer2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2310Importer2310.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2321Importer2321.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340Importer2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340Importer2341.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340Importer2342.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2341/Scan2341Importer2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2341/Scan2341Importer2341.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342Importer2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342Importer2342.hpp>
#include <microvision/common/sdk/datablocks/scan/ScanImporter2202.hpp>
#include <microvision/common/sdk/datablocks/scan/ScanImporter2205.hpp>
#include <microvision/common/sdk/datablocks/scan/ScanImporter2208.hpp>
#include <microvision/common/sdk/datablocks/scan/ScanImporter2209.hpp>
#include <microvision/common/sdk/datablocks/scan/ScanImporter2321.hpp>

#include <microvision/common/sdk/datablocks/stateofoperation/special/StateOfOperation9110Importer9110.hpp>
#include <microvision/common/sdk/datablocks/stateofoperation/special/StateOfOperation9111Importer9111.hpp>
#include <microvision/common/sdk/datablocks/stateofoperation/StateOfOperationImporter9111.hpp>

#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringCanStatus6700Importer6700.hpp>
#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringDeviceStatus6701Importer6701.hpp>
#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringSystemStatus6705Importer6705.hpp>

#include <microvision/common/sdk/datablocks/timerecord/special/TimeRecord9000Importer9000.hpp>
#include <microvision/common/sdk/datablocks/timerecord/TimeRecordImporter9000.hpp>

#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationsList9010Importer9010.hpp>
#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationsList9011Importer9011.hpp>
#include <microvision/common/sdk/datablocks/timerelation/TimeRelationsListImporter9010.hpp>
#include <microvision/common/sdk/datablocks/timerelation/TimeRelationsListImporter9011.hpp>

#include <microvision/common/sdk/datablocks/trafficlight/special/TrafficLightStateList3600Importer3600.hpp>
#include <microvision/common/sdk/datablocks/trafficlight/TrafficLightStateListImporter3600.hpp>

#include <microvision/common/sdk/datablocks/eventtag/special/UserEventTag7010Importer7010.hpp>

#include <microvision/common/sdk/datablocks/vehiclecontrol/special/VehicleControl9100Importer9100.hpp>
#include <microvision/common/sdk/datablocks/vehiclecontrol/VehicleControlImporter9100.hpp>

#include <microvision/common/sdk/datablocks/vehiclerequest/special/VehicleRequest9105Importer9105.hpp>
#include <microvision/common/sdk/datablocks/vehiclerequest/VehicleRequestImporter9105.hpp>

#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2805Importer2805.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2806Importer2806.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2807Importer2807.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2808Importer2808.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2809Importer2809.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleStateImporter2805.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleStateImporter2806.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleStateImporter2807.hpp>

#include <microvision/common/sdk/datablocks/vehiclestate/VehicleStateImporter2808.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleStateImporter2809.hpp>

#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/ZoneOccupationListA000ImporterA000.hpp>
