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
//! \date Feb 7, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/Unconvertable.hpp>
#include <microvision/common/sdk/misc/ToHex.hpp>

#include <microvision/common/sdk/io.hpp>
#include <microvision/common/sdk/bufferIO.hpp>

#include <boost/functional/hash.hpp>

#include <ostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Identification of serialized MVIS SDK data containers (data block) by a type.
//!
//! This class is used for serialization of DataType.
//!
//! \note Serializations of MVIS SDK data containers are identified by their DataType.
//------------------------------------------------------------------------------
class DataTypeId : public microvision::common::sdk::ComparableUnconvertable<uint16_t>
{
public:
    //========================================
    //! \brief Ids of some DataTypes.
    //----------------------------------------
    enum DataType : uint16_t
    {
        DataType_Unknown = 0x0000U,
        //0x0001U used
        //0x1000U used
        //0x1001U used
        DataType_CanMessage1002 = 0x1002U, //!< A single can message that has been received via Ethernet.
        //0x1010U used
        DataType_FrameEndSeparator1100 = 0x1100U,

        DataType_Command2010 = 0x2010U,
        //0x2011U used
        //0x201AU used
        DataType_Reply2020 = 0x2020U,
        //0x2021U used
        //0x202AU used
        DataType_Notification2030 = 0x2030U, //!< Error and warning messages sent by MVIS LUX/Scala family sensors
        //0x20E0U used
        //0x2200U used
        //0x2201U used
        DataType_Scan2202 = 0x2202U, //!< scan data sent by MVIS LUX/Scala (before B2) family sensors
        //0x2203U used
        //0x2204U used
        DataType_Scan2205 = 0x2205U, //!< scan data sent by MVIS ECU devices
        //0x2206U used
        //0x2207U used
        DataType_Scan2208 = 0x2208U, //!< Scan data sent by Scala B2 and MVIS MiniLux sensors
        DataType_Scan2209 = 0x2209U, //!< Identical to 2205 except it can hold more than 65535 points
        //0x2210U used
        //0x2212U used
        //0x2213U used
        //0x2220U used
        DataType_ObjectList2221 = 0x2221U, //!< objects sent by MVIS LUX family sensors
        //0x2222U used
        //0x2223U used
        //0x2224U used
        DataType_ObjectList2225 = 0x2225U, //!< objects sent by MVIS ECU devices
        //0x222AU used
        //0x2230U used
        //0x2231U used
        //0x2232U used
        //0x2233U used
        //0x2234U used
        //0x2235U used
        //0x2238U used
        //0x2239U used
        //0x2250U used
        //0x2251U used
        //0x2260U used
        DataType_ObjectList2270 = 0x2270U, //!< send by Scala family sensors (starting with B2)
        DataType_ObjectList2271 = 0x2271U, //!< send by Scala family sensors (starting with B2)
        //0x2275U used
        DataType_ObjectList2280 = 0x2280U, //!< send by ECU devices
        DataType_ObjectList2281 = 0x2281U, //!< send by ECU devices
        DataType_ObjectList2290 = 0x2290U, //!< generate by the Evaluation Suite
        DataType_ObjectList2291 = 0x2291U, //!< generate by the Evaluation Suite
        //0x2300U used
        //0x2301U used
        DataType_Scan2310 = 0x2310U, //! Uninterpreted Scala raw data from the FPGA
        //0x2320U used
        DataType_Scan2321 = 0x2321U, //! Lidar Scan in ThirdPartyVLidar raw format
        //0x2330U used
        DataType_RadarScan2331 = 0x2331U, //!< Radar Scan
        DataType_Scan2340      = 0x2340U, //!< used for MOVIA scan
        DataType_Scan2341      = 0x2341U, //!< used for low bandwidth MOVIA scan
        DataType_Scan2342      = 0x2342U, //!< used for low bandwidth MOVIA scan
        //0x2350U used
        //0x2351U used
        DataType_LdmiRawFrame2352        = 0x2352U, //!< used for MOVIA, LDMI raw with LDMIA200 as static info
        DataType_LdmiRawFrame2353        = 0x2353U, //!< used for MOVIA, LDMI raw with LDMIA300 as static info
        DataType_LdmiRawFrame2354        = 0x2354U, //!< used for MOVIA, LDMI raw with additional data
        DataType_LdmiAggregatedFrame2355 = 0x2355U, //!< used for MOVIA or ECU, complete LDMI raw frame
        DataType_MavinRawFrame2360       = 0x2360U, //!< used for MicroVision MAVIN LiDAR raw frame
        //0x2400U used
        //0x2401U used
        //0x2402U used
        DataType_Image2403 = 0x2403U, //!< An image
        DataType_Image2404 = 0x2404U, //!< Another image
        DataType_Image2405 = 0x2405U, //!< Yet Another image, extends 0x2404.
        //0x2600U used
        //0x2601U used
        //0x2602U used
        //0x2603U used
        DataType_PositionWgs84_2604 = 0x2604U, //!< GPS position
        DataType_OGpsImuMessage2610 = 0x2610U, //!<
        DataType_OGpsImuStatus2611  = 0x2611U, //!<
        //0x2612U used
        //0x2613U used
        //0x2614U used
        //0x2700U used
        //0x2800U used
        //0x2801U used
        //0x2802U used
        //0x2803U used
        //0x2804U used
        DataType_VehicleStateBasic2805 = 0x2805U, //!< send by LUX/Scala
        DataType_VehicleStateBasic2806 = 0x2806U, //!< send by ECU
        DataType_VehicleStateBasic2807 = 0x2807U, //!< send by ECU
        DataType_VehicleStateBasic2808 = 0x2808U, //!< send by ECU
        DataType_VehicleStateBasic2809 = 0x2809U, //!< 3d extension for 2808
        //0x2810U used
        //0x2820U used
        DataType_MeasurementList2821 = 0x2821U, //!< Data type that contains a single measurement list.
        //0x2830U used

        //0x3000U used
        //0x3001U used
        //0x3002U used
        //0x3003U used
        //0x3010U used
        //0x3011U used
        //0x3012U used
        //0x3050U used
        DataType_IdSequence3500            = 0x3500U,
        DataType_PositionWgs84Sequence3510 = 0x3510U,
        DataType_Destination3520           = 0x3520U,
        DataType_Destination3521           = 0x3521U,
        DataType_MissionHandlingStatus3530 = 0x3530U, //!< Information about the state of mission handling module
        DataType_MissionResponse3540       = 0x3540U,
        DataType_TrafficLight3600          = 0x3600U,

        //0x4000U used
        DataType_ObjectAssociationList4001 = 0x4001U,
        //0x4100U used
        //0x4110U used
        //0x4111U used
        //0x4200U used

        //0x5000U used

        DataType_IdcTrailer6120    = 0x6120U, //!< Trailer Message in an IDC file
        DataType_FrameIndex6130    = 0x6130U, //!< Index over IDC file
        DataType_GeoFrameIndex6140 = 0x6140U, //!< Index for geo frames in a tiled map IDC file.
        DataType_GeoFrameStart6150 = 0x6150U, //!< Marks the start of a geo frame in a tiled map IDC file.
        //0x6200U used
        //0x6201U used
        //0x6300U used
        DataType_DeviceStatus6301 = 0x6301U,
        //0x6302U used
        DataType_DeviceStatus6303 = 0x6303U,
        DataType_DeviceStatus6320 = 0x6320U, //!< State for MVIS SyncBox
        DataType_LogError6400     = 0x6400U,
        DataType_LogWarning6410   = 0x6410U,
        //0x641FU used
        DataType_LogNote6420  = 0x6420U,
        DataType_LogDebug6430 = 0x6430U,
        //0x6500U used
        //0x6501U used
        //0x6502U used
        DataType_ObjectLabel6503                  = 0x6503U,
        DataType_SystemMonitoringCanStatus6700    = 0x6700U,
        DataType_SystemMonitoringDeviceStatus6701 = 0x6701U,
        DataType_SystemMonitoringSystemStatus6705 = 0x6705U,
        //0x6800U used
        //0x6801U used
        //0x6802U used
        //0x6803U used
        //0x6804U used
        //0x6805U used
        //0x6806U used
        //0x6807U used
        //0x6808U used
        //0x6810U used
        //0x6811U used
        //0x6812U used
        //0x6813U used
        //0x6814U used
        //0x6815U used
        //0x6816U used
        DataType_LogPolygonList2dFloat6817 = 0x6817U, //!< List of informational polygons with text label
        //0x6818U used
        DataType_Marker6820 = 0x6820U, //!< Marker message for various purpose, e.g. debugging
        //0x6900U used
        DataType_LaneMarkingList6901  = 0x6901U, //! Used for generating Lanes.
        DataType_RoadBoundaryList6902 = 0x6902U, //! Used for generating road Boundaries. type is identical to 0x6901
        //0x6910U used
        //0x6911U used
        //0x6912U used
        //0x6920U used
        //0x6930U used
        //0x6940U used
        //0x6941U used
        //0x6950U used
        //0x6961U used
        DataType_CarriageWayList6970 = 0x6970U, //!< Basic CarriageWayList
        //0x6971U used
        DataType_CarriageWayList6972 = 0x6972U, //!< CarriageWayList with additional LaneSegment marking properties
        //0x6980U used
        //0x6990U used

        DataType_EventTag7000            = 0x7000U,
        DataType_EventMarker7001         = 0x7001U,
        DataType_UserEventTag7010        = 0x7010U,
        DataType_ContentSeparator7100    = 0x7100U,
        DataType_MetaInformationList7110 = 0x7110U,
        DataType_PointCloud7500          = 0x7500U, //!< PointCloudGlobal
        DataType_PointCloud7510          = 0x7510U, //!< PointCloudPlane
        DataType_PointCloud7511          = 0x7511U, //!< PointCloudPlane with tile offset

        //0x8000U used

        DataType_TimeRecord9000        = 0x9000U,
        DataType_GpsImu9001            = 0x9001U,
        DataType_Odometry9002          = 0x9002U,
        DataType_Odometry9003          = 0x9003U,
        DataType_GpsImu9004            = 0x9004U,
        DataType_TimeRelationsList9010 = 0x9010U, //!< Time Relations
        DataType_TimeRelationsList9011 = 0x9011U, //!< Time Relations9011
        DataType_VehicleControl9100    = 0x9100U,
        DataType_VehicleRequest9105    = 0x9105U,
        DataType_StateOfOperation9110  = 0x9110U,
        DataType_StateOfOperation9111  = 0x9111U,
        //0x9120U used
        //0x9130U used
        //0x9140U used
        //0x9150U used
        //0x9152U used
        //0x9154U used
        //0x9156U used
        //0x9158U used
        //0x9160U used

        DataType_ZoneOccupationListA000 = 0xA000U,

        DataType_CustomDataContainer = 0xFFFEU, //!< special handled custom data container containing a uuid to identify
        DataType_LastId              = 0xFFFFU
    }; // DataType

public:
    //========================================
    //! \brief Explicit constructor of DataTypeId.
    //!
    //! \param[in] dtId  DataType Id as integer.
    //----------------------------------------
    explicit DataTypeId(const uint16_t dtId) : microvision::common::sdk::ComparableUnconvertable<uint16_t>(dtId) {}

    //========================================
    //! \brief Constructor
    //!
    //! \param[in] dt  DataType Id.
    //----------------------------------------
    DataTypeId(const DataType dt) : microvision::common::sdk::ComparableUnconvertable<uint16_t>(uint16_t(dt)) {}

    //========================================
    //! \brief Constructor
    //!
    //! Id will be set as unknown.
    //----------------------------------------
    DataTypeId() : microvision::common::sdk::ComparableUnconvertable<uint16_t>(uint16_t(DataType_Unknown)) {}

public:
    bool isset() const { return (this->m_data != DataType_Unknown); }
    void unset() { this->m_data = uint16_t(DataType_Unknown); }

public:
    static std::streamsize getSerializedSize() { return sizeof(uint16_t); }

public:
    std::istream& readBE(std::istream& is)
    {
        microvision::common::sdk::readBE(is, this->m_data);
        return is;
    }

    std::ostream& writeBE(std::ostream& os) const
    {
        microvision::common::sdk::writeBE(os, this->m_data);
        return os;
    }

    void readBE(const char*& target) { microvision::common::sdk::readBE(target, this->m_data); }

    void writeBE(char*& target) const { microvision::common::sdk::writeBE(target, this->m_data); }
}; // DataTypeId

//==============================================================================

template<>
inline std::string toHex<DataTypeId>(const DataTypeId t)
{
    return toHex(uint16_t(t));
}

//==============================================================================

inline std::ostream& operator<<(std::ostream& os, const DataTypeId dataTypeId)
{
    os << toHex(dataTypeId);
    return os;
}

//==============================================================================

inline std::ostream& operator<<(std::ostream& os, const DataTypeId::DataType dataTypeId)
{
    os << toHex(DataTypeId{dataTypeId});
    return os;
}

///! \brief Convert DataTypeId::DataType to std::string
std::string datatypeToString(const DataTypeId datatype);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
namespace boost {
//==============================================================================

//==============================================================================
//! \brief The template specializations of boost::hash for \c DataTypeId.
//! \deprecated Please use std::hash and in implicit use,
//!             for example std::unordered_map instead of boost::unordered_map.
//------------------------------------------------------------------------------
template<>
struct MICROVISION_SDK_DEPRECATED hash<microvision::common::sdk::DataTypeId>
{
    //========================================
    //! \brief Create hash of \c DataTypeId.
    //----------------------------------------
    std::size_t operator()(microvision::common::sdk::DataTypeId const& dataTypeId) const
    {
        hash<uint16_t> hasher;
        return hasher(dataTypeId);
    }
}; // :hash<DataTypeId>

//==============================================================================
} // namespace boost
//==============================================================================

//==============================================================================
namespace std {
//==============================================================================

//==============================================================================
//! \brief The template specializations of std::hash for \c DataTypeId.
//------------------------------------------------------------------------------
template<>
struct hash<microvision::common::sdk::DataTypeId>
{
    //========================================
    //! \brief Create hash of \c DataTypeId.
    //----------------------------------------
    std::size_t operator()(microvision::common::sdk::DataTypeId const& dataTypeId) const
    {
        hash<uint16_t> hasher;
        return hasher(dataTypeId);
    }
}; // :hash<DataTypeId>

//==============================================================================
} // namespace std
//==============================================================================
