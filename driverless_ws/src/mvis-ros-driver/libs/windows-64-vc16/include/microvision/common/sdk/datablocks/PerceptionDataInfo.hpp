//==============================================================================
//! \file
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/ScannerType.hpp>

#include <microvision/common/sdk/NtpTime.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief A snippet used in some data types to hold some meta information
//!        for the data.
//!
//! A PerceptionDataInfo contains measurement timestamp, used coordinate system,
//! sequential and device ID as well as the data origin, i.e. the type of device or
//! sensor used.
//------------------------------------------------------------------------------
class PerceptionDataInfo final
{
public:
    //========================================
    //! \brief Serialized size of an PerceptionDataInfo.
    //----------------------------------------
    static constexpr uint8_t serializedSize{23};

    //========================================
    //! \brief Serialized size of an PerceptionDataInfo for for (internal) tool generated datatype version.
    //----------------------------------------
    static constexpr uint8_t serializedSizeEA{31};

public:
    //========================================
    //! \brief Enum specifying the coordinate frame.
    //----------------------------------------
    enum class CoordinateFrame : uint8_t
    {
        WorldReference             = 0, //!< World reference coordinate system.
        VehicleRoad                = 1, //!< Vehicle road coordinate system.
        VehicleBody                = 2, //!< Vehicle body coordinate system.
        SensorHousing              = 3, //!< Sensor housing coordinate system.
        SensorMeasurement          = 4, //!< Sensor measurement coordinate system.
        SensorAdjustment           = 5, //!< Sensor adjustment coordinate system.
        IdealizedSensorMeasurement = 6 //!< Idealized sensor measurement coordinate system.
    };

    //========================================
    //! \brief Enum specifying the sensor/device type.
    //----------------------------------------
    enum class DataOrigin : uint16_t
    {
        Can                 = 0, //!< Generic CAN bus device.
        XsenseImu           = 1,
        GenesysAdma         = 2,
        ThirdPartyOGpsImu   = 3,
        Ublox               = 4,
        Lux                 = 30, //!< MVIS LUX lidar sensor.
        Scala               = 31, //!< Scala lidar sensor.
        Lms                 = 32, //!< Thirdparty LMS lidar sensor.
        Movia               = 33, //!< MOVIA sensor.
        Timer               = 34,
        ThirdPartyVLidar    = 35, //!< Third party lidar sensor.
        Mavin               = 36, //!< MicroVision MAVIN sensor
        ThirdPartyRLidar    = 37, //!< Third party lidar sensor.
        Camera              = 50, //!< Generic Camera bus device.
        Radar               = 59, //!< deprecated
        RadarAstyx          = 60,
        RadarAc2000         = 61,
        RadarArs408         = 62,
        RadarAc3000         = 63,
        RadarHella5ga1      = 64,
        DynamicObjectExpert = 70,
        StaticObjectExpert  = 71,
        EgoMotionExpert     = 72,
        UltrasonicGeneral   = 90, //!< General ultrasonic device.
        UltrasonicBoschDc1  = 91,
        UltrasonicHfm       = 92,
        Simulation          = 100 //!< The data origin is a simulation.
    }; // DeviceType

public:
    //========================================
    //! \brief Get the serialized size of an PerceptionDataInfo.
    //! \return Return the serialized size of an PerceptionDataInfo.
    //----------------------------------------
    static constexpr std::streamsize getSerializedSize_static() { return serializedSize; }

    //========================================
    //! \brief Get the serialized size of an PerceptionDataInfo for EA version.
    //! \return Return the serialized size of an PerceptionDataInfo for EA version.
    //----------------------------------------
    static constexpr std::streamsize getSerializedSizeEA_static() { return serializedSizeEA; }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    PerceptionDataInfo() = default;

    //========================================
    //! \brief Constructor.
    //! \param[in] sequenceId               The (per data source) unique sequence ID.
    //! \param[in] creationSystemTimestamp  The creation system time stamp.
    //! \param[in] coordinateFrame          The coordinate frame/system.
    //! \param[in] dataOrigin               The data origin ("sensor" type).
    //! \param[in] deviceId                 The device ID.
    //----------------------------------------
    PerceptionDataInfo(const uint32_t sequenceId,
                       const NtpTime creationSystemTimestamp,
                       const CoordinateFrame coordinateFrame,
                       const DataOrigin dataOrigin,
                       const uint64_t deviceId);

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~PerceptionDataInfo() = default;

    PerceptionDataInfo(const PerceptionDataInfo& other) = default;
    PerceptionDataInfo& operator=(const PerceptionDataInfo& other) = default;

public:
    //========================================
    //! \brief Deserialize an PerceptionDataInfo in deprecated ABI from \a is.
    //! \param[in, out] is  An input stream containing the serialized
    //!                     data of the PerceptionDataInfo to
    //!                     be read.
    //! \return \c True if the expected number of bytes have been read
    //!         without an error. \c false otherwise.
    //! \sa getSerializedSize
    //! \sa serialize
    //----------------------------------------
    bool deserialize(std::istream& is);

    //========================================
    //! \brief Serialize this PerceptionDataInfo in deprecated ABI to \a os.
    //! \param[in, out] os  An output stream this object shall be
    //!                     written to.
    //! \return \c True if the expected number of bytes have been written
    //!         without an error. \c false otherwise.
    //! \sa getSerializedSize
    //! \sa deserialize
    //----------------------------------------
    bool serialize(std::ostream& os) const;

    //========================================
    //! \brief Deserialize an PerceptionDataInfo from \a is.
    //! \param[in, out] is  An input stream containing the serialized
    //!                     data of the PerceptionDataInfo to
    //!                     be read.
    //! \return \c True if the expected number of bytes have been read
    //!         without an error. \c false otherwise.
    //! \sa getSerializedSize
    //! \sa serialize
    //----------------------------------------
    bool deserializeEA(std::istream& is);

    //========================================
    //! \brief Serialize this PerceptionDataInfo to \a os.
    //! \param[in, out] os  An output stream this object shall be
    //!                     written to.
    //! \return \c True if the expected number of bytes have been written
    //!         without an error. \c false otherwise.
    //! \sa getSerializedSize
    //! \sa deserialize
    //----------------------------------------
    bool serializeEA(std::ostream& os) const;

    //========================================
    //! \brief Get the serialized size of this PerceptionDataInfo.
    //! \return Return the serialized size of this PerceptionDataInfo.
    //----------------------------------------
    std::streamsize getSerializedSize() const { return serializedSize; }

public: // getter
    //========================================
    //! \brief Get the sequence ID.
    //! \return Return the sequence ID.
    //----------------------------------------
    uint32_t getSequenceId() const { return m_sequenceId; }

    //========================================
    //! \brief Get the creation system time stamp.
    //! \return Return the creation system time stamp.
    //----------------------------------------
    NtpTime getCreationSystemTimestamp() const { return m_creationSystemTimestamp; }

    //========================================
    //! \brief Get the coordinate frame.
    //! \return Return the coordinate frame.
    //----------------------------------------
    CoordinateFrame getCoordinateFrame() const { return m_coordinateFrame; }

    //========================================
    //! \brief Get the data origin (device type).
    //! \return Return the data origin (device type).
    //----------------------------------------
    DataOrigin getDataOrigin() const { return m_dataOrigin; }

    //========================================
    //! \brief Get the data origin (device/sensor type) as ScannerType.
    //! \return Return the data origin (device/sensor type).
    //----------------------------------------
    ScannerType getScannerType() const;

    //========================================
    //! \brief Get the device ID.
    //! \return Return the device ID.
    //----------------------------------------
    uint64_t getDeviceId() const { return m_deviceId; }

public: // setter
    //========================================
    //! \brief Set the sequence ID.
    //! param]in] newSequenceId  The new sequence ID.
    //----------------------------------------
    void setSequenceId(const uint32_t newSequenceId) { m_sequenceId = newSequenceId; }

    //========================================
    //! \brief Get the creation system time stamp.
    //! \param[in] newCreationSystemTimestamp  The new creation system time stamp.
    //----------------------------------------
    void setCreationSystemTimestamp(const NtpTime newCreationSystemTimestamp)
    {
        m_creationSystemTimestamp = newCreationSystemTimestamp;
    }

    //========================================
    //! \brief Get the coordinate frame.
    //! \param[in] newCoordinateFrame  The new coordinate frame.
    //----------------------------------------
    void setCoordinateFrame(const CoordinateFrame newCoordinateFrame) { m_coordinateFrame = newCoordinateFrame; }

    //========================================
    //! \brief Get the data origin (device/sensor type).
    //! \param[in]  newDataOrigin  The new data origin (device/sensor type).
    //----------------------------------------
    void setDataOrigin(const DataOrigin newDataOrigin) { m_dataOrigin = newDataOrigin; }

    //========================================
    //! \brief Get the data origin (device/sensor type).
    //! \param[in]  newDataOrigin  The new data origin (device/sensor type).
    //----------------------------------------
    void setDataOrigin(const ScannerType newDataOrigin);

    //========================================
    //! \brief Get the device ID.
    //! \param[in] newDeviceId  The new device ID.
    //----------------------------------------
    void setDeviceId(const uint64_t newDeviceId) { m_deviceId = newDeviceId; }

private:
    //========================================
    //! \brief Uniquely identifies a datatype object.
    //----------------------------------------
    uint32_t m_sequenceId{0};

    //========================================
    //! \brief NTP timestamp of datatype creation
    //----------------------------------------
    NtpTime m_creationSystemTimestamp{};

    //========================================
    //! \brief The coordinate system of this datatype object.
    //----------------------------------------
    CoordinateFrame m_coordinateFrame{CoordinateFrame::VehicleBody};

    //========================================
    //! \brief The sensor/device type.
    //----------------------------------------
    DataOrigin m_dataOrigin{DataOrigin::Can};

    //========================================
    //! \brief The device ID.
    //----------------------------------------
    uint64_t m_deviceId{0};
}; // PerceptionDataInfo

//==============================================================================
//==============================================================================

template<>
inline void readLE<PerceptionDataInfo::CoordinateFrame>(std::istream& is, PerceptionDataInfo::CoordinateFrame& value)
{
    using EnumIntType = std::underlying_type<PerceptionDataInfo::CoordinateFrame>::type;
    EnumIntType tmp;
    readLE(is, tmp);
    value = static_cast<PerceptionDataInfo::CoordinateFrame>(tmp);
}

//==============================================================================

template<>
inline void writeLE<PerceptionDataInfo::CoordinateFrame>(std::ostream& os,
                                                         const PerceptionDataInfo::CoordinateFrame& value)
{
    using EnumIntType = std::underlying_type<PerceptionDataInfo::CoordinateFrame>::type;
    const EnumIntType tmp{static_cast<EnumIntType>(value)};
    writeLE(os, tmp);
}

//==============================================================================

template<>
inline void readBE<PerceptionDataInfo::CoordinateFrame>(std::istream& is, PerceptionDataInfo::CoordinateFrame& value)
{
    using EnumIntType = std::underlying_type<PerceptionDataInfo::CoordinateFrame>::type;
    EnumIntType tmp;
    readBE(is, tmp);
    value = static_cast<PerceptionDataInfo::CoordinateFrame>(tmp);
}

//==============================================================================

template<>
inline void writeBE<PerceptionDataInfo::CoordinateFrame>(std::ostream& os,
                                                         const PerceptionDataInfo::CoordinateFrame& value)
{
    using EnumIntType = std::underlying_type<PerceptionDataInfo::CoordinateFrame>::type;
    const EnumIntType tmp{static_cast<EnumIntType>(value)};
    writeBE(os, tmp);
}

//==============================================================================

template<>
inline void readLE<PerceptionDataInfo::DataOrigin>(std::istream& is, PerceptionDataInfo::DataOrigin& value)
{
    using EnumIntType = std::underlying_type<PerceptionDataInfo::DataOrigin>::type;
    EnumIntType tmp;
    readLE(is, tmp);
    value = static_cast<PerceptionDataInfo::DataOrigin>(tmp);
}

//==============================================================================

template<>
inline void writeLE<PerceptionDataInfo::DataOrigin>(std::ostream& os, const PerceptionDataInfo::DataOrigin& value)
{
    using EnumIntType = std::underlying_type<PerceptionDataInfo::DataOrigin>::type;
    const EnumIntType tmp{static_cast<EnumIntType>(value)};
    writeLE(os, tmp);
}

//==============================================================================

template<>
inline void readBE<PerceptionDataInfo::DataOrigin>(std::istream& is, PerceptionDataInfo::DataOrigin& value)
{
    using EnumIntType = std::underlying_type<PerceptionDataInfo::DataOrigin>::type;
    EnumIntType tmp;
    readBE(is, tmp);
    value = static_cast<PerceptionDataInfo::DataOrigin>(tmp);
}

//==============================================================================

template<>
inline void writeBE<PerceptionDataInfo::DataOrigin>(std::ostream& os, const PerceptionDataInfo::DataOrigin& value)
{
    using EnumIntType = std::underlying_type<PerceptionDataInfo::DataOrigin>::type;
    const EnumIntType tmp{static_cast<EnumIntType>(value)};
    writeBE(os, tmp);
}

//==============================================================================
// operators
//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const PerceptionDataInfo& lhs, const PerceptionDataInfo& rhs);

//==============================================================================

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const PerceptionDataInfo& lhs, const PerceptionDataInfo& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
