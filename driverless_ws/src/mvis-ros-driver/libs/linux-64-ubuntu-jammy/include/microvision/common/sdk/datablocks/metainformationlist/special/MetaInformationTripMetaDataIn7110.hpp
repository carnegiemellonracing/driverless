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
//! \date Nov 05, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationList7110.hpp>
#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationBaseIn7110.hpp>

#include <boost/any.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief
//!
//! This class contains various properties holding information further describing
//! the circumstances of the recording for this trip. For example vehicle, driver, environment ...
//!
//! One trip could be a collection of various recordings (idc files for example).
//------------------------------------------------------------------------------
class MetaInformationTripMetaDataIn7110 : public MetaInformationBaseIn7110
{
public:
    using MetaInformationTripMetaDataIn7110SPtr = std::shared_ptr<MetaInformationTripMetaDataIn7110>;

public:
    //========================================
    //! \brief The key of the property.
    //----------------------------------------
    enum class PropertyKey : uint16_t
    {
        Undefined                   = 0, //!< Property key not defined.
        TripStartTimestamp          = 1, //!< Date and time when the trip was started.
        VehicleInternalName         = 2, //!< Customer specific name of the vehicle.
        VehicleModel                = 3, //!< Name of the manufacturer and model of the vehicle.
        LicensePlate                = 4, //!< License plate number of the vehicle.
        VehicleIdentificationNumber = 5, //!< Identification number of the vehicle.
        CalibrationDate             = 6, //!< Most recent date when the sensors were calibrated.
        CalibrationUserName         = 7, //!< Name of the person who did the last calibration.
        DriverName                  = 8, //!< Name of the driver on this trip.
        CoDriverName                = 9, //!< Name of the co-driver on this trip.
        CustomerName                = 10, //!< Name of the customer who ordered this trip.
        ProjectName                 = 11, //!< Project name and / or number this trip belongs to.
        StreetType                  = 12, //!< Type of street used mainly during this trip.
        Tags                        = 13 //!< List of tags describing the trip separated by semicolons.
    }; // PropertyKey

    //========================================
    //! \brief The type of the property.
    //----------------------------------------
    enum class PropertyType : uint8_t
    {
        Undefined = 0,
        Float     = 1,
        Double    = 2,
        Int8      = 3,
        UInt8     = 4,
        Int16     = 5,
        UInt16    = 6,
        Int32     = 7,
        UInt32    = 8,
        Int64     = 9,
        UInt64    = 10,
        Bool      = 11,
        String    = 12,
        NtpTime   = 13
    }; // PropertyType

public:
    //========================================
    //! \brief Constructor from type.
    //!
    //! Initializes this instance as meta information of type config file.
    //----------------------------------------
    MetaInformationTripMetaDataIn7110()
      : MetaInformationBaseIn7110(MetaInformationBaseIn7110::MetaInformationType::TripMetaData)
    {}

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    ~MetaInformationTripMetaDataIn7110() override = default;

public: // high-level getter
    //========================================
    //! \brief Get the date and time when the trip was started.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The date and time when the trip was started.
    //----------------------------------------
    static NtpTime getTripStartTimestamp(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the customer specific name of the vehicle.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The customer specific name of the vehicle.
    //----------------------------------------
    static std::string getVehicleInternalName(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the name of the manufacturer and model of the vehicle.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The name of the manufacturer and model of the vehicle.
    //----------------------------------------
    static std::string getVehicleModel(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the license plate number of the vehicle.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The license plate number of the vehicle.
    //----------------------------------------
    static std::string getLicensePlate(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the identification number of the vehicle.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The identification number of the vehicle.
    //----------------------------------------
    static std::string getVehicleIdentificationNumber(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the most recent date when the sensors were calibrated.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The most recent date when the sensors were calibrated.
    //----------------------------------------
    static NtpTime getCalibrationDate(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the name of the person who did the last calibration.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The name of the person who did the last calibration.
    //----------------------------------------
    static std::string getCalibrationUserName(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the name of the driver on this trip.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The name of the driver on this trip.
    //----------------------------------------
    static std::string getDriverName(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the name of the co-driver on this trip.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The name of the co-driver on this trip.
    //----------------------------------------
    static std::string getCoDriverName(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the name of the customer who ordered this trip.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The name of the customer who ordered this trip.
    //----------------------------------------
    static std::string getCustomerName(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the project name and / or number this trip belongs to.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The project name and / or number this trip belongs to.
    //----------------------------------------
    static std::string getProjectName(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the type of street used mainly during this trip.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The type of street used mainly during this trip.
    //----------------------------------------
    static std::string getStreetType(const MetaInformationList7110& mil);

    //========================================
    //! \brief Get the list of tags describing the trip.
    //!
    //! \param[in] mil  List with meta information to search for this data.
    //! \return The list of tags describing the trip.
    //!
    //! \note The individual tags are separated by semicolons.
    //----------------------------------------
    static std::string getTags(const MetaInformationList7110& mil);

public: // high-level setter
    //========================================
    //! \brief Set the date and time when the trip was started.
    //!
    //! \param[in,out] mil                 List with meta information to add this data to.
    //! \param[in]     tripStartTimestamp  The date and time when the trip was started.
    //----------------------------------------
    static void setTripStartTimestamp(MetaInformationList7110& mil, const NtpTime tripStartTimestamp);

    //========================================
    //! \brief Set the customer specific name of the vehicle.
    //!
    //! \param[in,out] mil                  List with meta information to add this data to.
    //! \param[in]     vehicleInternalName  The customer specific name of the vehicle.
    //----------------------------------------
    static void setVehicleInternalName(MetaInformationList7110& mil, const std::string& vehicleInternalName);

    //========================================
    //! \brief Set the name of the manufacturer and model of the vehicle.
    //!
    //! \param[in,out] mil           List with meta information to add this data to.
    //! \param[in]     vehicleModel  The name of the manufacturer and model of the vehicle.
    //----------------------------------------
    static void setVehicleModel(MetaInformationList7110& mil, const std::string& vehicleModel);

    //========================================
    //! \brief Set the license plate number of the vehicle.
    //!
    //! \param[in,out] mil           List with meta information to add this data to.
    //! \param[in]     licensePlate  The license plate number of the vehicle.
    //----------------------------------------
    static void setLicensePlate(MetaInformationList7110& mil, const std::string& licensePlate);

    //========================================
    //! \brief Set the identification number of the vehicle.
    //!
    //! \param[in,out] mil                          List with meta information to add this data to.
    //! \param[in]     vehicleIdentificationNumber  The identification number of the vehicle.
    //----------------------------------------
    static void setVehicleIdentificationNumber(MetaInformationList7110& mil,
                                               const std::string& vehicleIdentificationNumber);

    //========================================
    //! \brief Set the most recent date when the sensors were calibrated.
    //!
    //! \param[in,out] mil              List with meta information to add this data to.
    //! \param[in]     calibrationDate  The most recent date when the sensors were calibrated.
    //----------------------------------------
    static void setCalibrationDate(MetaInformationList7110& mil, const NtpTime calibrationDate);

    //========================================
    //! \brief Set the name of the person who did the last calibration.
    //!
    //! \param[in,out] mil                  List with meta information to add this data to.
    //! \param[in]     calibrationUserName  The name of the person who did the last calibration.
    //----------------------------------------
    static void setCalibrationUserName(MetaInformationList7110& mil, const std::string& calibrationUserName);

    //========================================
    //! \brief Set the name of the driver on this trip.
    //!
    //! \param[in,out] mil         List with meta information to add this data to.
    //! \param[in]     driverName  The name of the driver on this trip.
    //----------------------------------------
    static void setDriverName(MetaInformationList7110& mil, const std::string& driverName);

    //========================================
    //! \brief Set the name of the co-driver on this trip.
    //!
    //! \param[in,out] mil           List with meta information to add this data to.
    //! \param[in]     coDriverName  The name of the co-driver on this trip.
    //----------------------------------------
    static void setCoDriverName(MetaInformationList7110& mil, const std::string& coDriverName);

    //========================================
    //! \brief Set the name of the customer who ordered this trip.
    //!
    //! \param[in,out] mil           List with meta information to add this data to.
    //! \param[in]     customerName  The name of the customer who ordered this trip.
    //----------------------------------------
    static void setCustomerName(MetaInformationList7110& mil, const std::string& customerName);

    //========================================
    //! \brief Set the project name and / or number this trip belongs to.
    //!
    //! \param[in,out] mil          List with meta information to add this data to.
    //! \param[in]     projectName  The project name and / or number this trip belongs to.
    //----------------------------------------
    static void setProjectName(MetaInformationList7110& mil, const std::string& projectName);

    //========================================
    //! \brief Set the type of street used mainly during this trip.
    //!
    //! \param[in,out] mil         List with meta information to add this data to.
    //! \param[in]     streetType  The type of street used mainly during this trip.
    //----------------------------------------
    static void setStreetType(MetaInformationList7110& mil, const std::string& streetType);

    //========================================
    //! \brief Set the list of tags describing the trip.
    //!
    //! \param[in,out] mil   List with meta information to add this data to.
    //! \param[in]     tags  The list of tags describing the trip.
    //!
    //! \note The individual tags are separated by semicolons.
    //----------------------------------------
    static void setTags(MetaInformationList7110& mil, const std::string& tags);

public: // low-level getter
    //========================================
    //! \brief Get the key of this property.
    //!
    //! \return The key of this property.
    //----------------------------------------
    PropertyKey getKey() const { return m_key; }

    //========================================
    //! \brief Get the value of this property.
    //!
    //! \tparam T Type of the property value.
    //! \return The value of this property.
    //!
    //! \note If the value cannot be casted to the given type a \c boost::bad_any_cast exception is thrown.
    //----------------------------------------
    template<typename T>
    T getValue() const
    {
        return boost::any_cast<T>(m_data);
    }

    //========================================
    //! \brief Get the value of this property.
    //!
    //! \tparam    T             Type of the property value.
    //! \param[in] defaultValue  The value that is used if the property cannot be casted to the given type.
    //! \return The value of this property or the given default value if the property cannot be casted
    //!         to the given type.
    //----------------------------------------
    template<typename T>
    T getValue(const T& defaultValue) const
    {
        T result;
        try
        {
            result = boost::any_cast<T>(m_data);
        }
        catch (const boost::bad_any_cast&)
        {
            result = defaultValue;
        }

        return result;
    }

public: // low-level setter
    //========================================
    //! \brief Set the key of this property.
    //!
    //! \param[in] key  The new key of this property.
    //----------------------------------------
    void setKey(const PropertyKey key) { m_key = key; }

    //========================================
    //! \brief Set the value of this property.
    //!
    //! \tparam    T      Type of the property value.
    //! \param[in] value  The new value of this property.
    //----------------------------------------
    template<typename T>
    void setValue(const T& data)
    {
        m_data = data;
    }

    //========================================
    //! \brief Set both the key and the value of this property.
    //!
    //! \tparam    T      Type of the property value.
    //! \param[in] key    The new key of this property.
    //! \param[in] value  The new value of this property.
    //----------------------------------------
    template<typename T>
    void setValue(const PropertyKey key, const T& value)
    {
        setKey(key);
        setValue(value);
    }

    //========================================
    //! \brief Clear the value of this property.
    //----------------------------------------
    void resetValue() { m_data = boost::any(); }

public:
    //========================================
    //! \brief Check if the property value is of the given type.
    //!
    //! \tparam T  Type of the property value to check.
    //! \return \c True, if the property value is of the given type, \c false otherwise.
    //----------------------------------------
    template<typename T>
    bool isOfType() const
    {
        return m_data.type() == typeid(T);
    }

public:
    //========================================
    //! \brief Tests this meta information for equality.
    //!
    //! \param[in] otherBase  The other meta information to compare with.
    //! \return \c True, if the two meta information are equal, \c false otherwise.
    //----------------------------------------
    bool isEqual(const MetaInformationBaseIn7110& otherBase) const override;

    //========================================
    //! \brief Get the size of the serialized payload.
    //!
    //! \return The size of the serialized payload.
    //----------------------------------------
    uint32_t getSerializedPayloadSize() const override;

    //========================================
    //! \brief Deserialize the data from a stream.
    //!
    //! \param[in,out] is           The stream to read the data from.
    //! \param[in]     payloadSize  The size of the payload in the stream.
    //! \return \c True, if the deserialization was successful, \c false otherwise.
    //----------------------------------------
    bool deserializePayload(std::istream& is, const uint32_t payloadSize) override;

    //========================================
    //! \brief Serialize the data into a stream.
    //!
    //! \param[in,out] os           The stream to write the data to.
    //! \return \c True, if the serialization was successful, \c false otherwise.
    //----------------------------------------
    bool serializePayload(std::ostream& os) const override;

private:
    //========================================
    //! \brief Convert a type info to a property type.
    //!
    //! \param[in] typeInfo  Type info to convert.
    //! \return The corresponding property type or \a PropertyType::Undefined if the type info is not valid.
    //----------------------------------------
    static PropertyType typeInfoToType(const std::type_info& typeInfo);

    //========================================
    //! \brief Get a property value of a given type from a meta information list.
    //!
    //! \tparam    T             The type of property value to be returned.
    //! \param[in] mil           The meta information list to search.
    //! \param[in] propertyKey   The property kep to get the value for.
    //! \param[in] defaultValue  The value that is used if the property is not in the list.
    //! \return The property value or the \a defaultValue if the property was not found.
    //----------------------------------------
    template<typename T>
    static T getPropertyValue(const MetaInformationList7110& mil, const PropertyKey propertyKey, const T& defaultValue);

    //========================================
    //! \brief Set a property value of a given type in a meta information list.
    //!
    //! \tparam    T            The type of property value to be set.
    //! \param[in] mil          The meta information list to modify.
    //! \param[in] propertyKey  The property kep to set the value for.
    //! \param[in] value        The value to be set.
    //----------------------------------------
    template<typename T>
    static void setPropertyValue(MetaInformationList7110& mil, const PropertyKey propertyKey, const T& value);

    //========================================
    //! \brief Get the size of the property value for serialization.
    //!
    //! \return The size of the property value for serialization.
    //----------------------------------------
    std::size_t getDataSize() const;

private:
    PropertyKey m_key{PropertyKey::Undefined};
    boost::any m_data;

    static constexpr const char* loggerId = "microvision::common::sdk::MetaInformationList7110";
    static microvision::common::logging::LoggerSPtr logger;
}; // MetaInformationTripMetaDataIn7110

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
