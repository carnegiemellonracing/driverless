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
//! \date Jan 22, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/special/Version448In6303.hpp>
#include <microvision/common/sdk/datablocks/devicestatus/special/SerialNumberIn6303.hpp>
#include <microvision/common/sdk/datablocks/scan/ScannerInfo.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Current draft of device status definition for Scala B2 >X100 (generic)
//!
//! General data type: \ref microvision::common::sdk::DeviceStatus
//------------------------------------------------------------------------------
class DeviceStatus6303 : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;
    //	friend class ::DeviceStatus6303Test;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.devicestatus6303"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Id of the content in the variable part of the data type.
    //----------------------------------------
    // Attention: This value will be saved as uint16_t, hence
    //            do not use ids greater than 0xFFFF
    enum class ContentId : uint16_t
    {
        Illegal                      = 0x0000U,
        ApdVoltageTable              = 0x0001U, // float
        NoiseTable                   = 0x0002U, // uint16_t
        AdaptiveApdVoltageNoiseArray = 0x0003U, // struct AdaptiveAPDVoltageAndNoiseInfo
        // reserved
        ScalaARMVersion = 0x0100U, // uint32_t
        // reserved
        ErrorScalaFPGA = 0x0110U, // uint32_t
        // reserved
        YawOffsetConf = 0x0120U, // uint16_t
        YawOffsetCalc = 0x0121U, // int16_t
        VelFactorConf = 0x0122U, // uint16_t
        VelFactorCalc = 0x0123U, // uint16_t

        Dummy1 = 0xFFF1U, // struct Dummy1
        Dummy2 = 0xFFF2U, // struct Dummy2
        Dummy3 = 0xFFF3U, // struct Dummy3
        Dummy4 = 0xFFF4U // struct Dummy4
        //		Dummy5          = 0xFFF5,  // struct Dummy5
        //		Dummy6          = 0xFFF6,  // struct Dummy6
        //		Dummy7          = 0xFFF7,  // struct Dummy7
        //		Dummy8          = 0xFFF8,  // struct Dummy8
        //		Dummy9          = 0xFFF9,  // struct Dummy9
        // reserved
    }; // ContentId

    //========================================
    //! \brief Type of the content.
    //!
    //! User defined structs has to be decoded by
    //! the user himself. DeviceStatus6303 does not
    //! know about that contents.
    //----------------------------------------
    enum class ElementType : uint8_t
    {
        Illegal = 0x00U,
        UINT8   = 0x01U, //!< Content consists of uint8_t
        INT8    = 0x02U, //!< Content consists of int8_t
        UINT16  = 0x03U, //!< Content consists of uint16_t
        INT16   = 0x04U, //!< Content consists of int16_t
        UINT32  = 0x05U, //!< Content consists of uint32_t
        INT32   = 0x06U, //!< Content consists of int32_t
        FLOAT32 = 0x07U, //!< Content consists of float
        STRUCT  = 0x08U //!< Content consists of user defined struct

        // reserved
    }; // ElementType

public:
    class ContentDescr;
    class ContentDescrDeserializer;
    class UserDefinedStructBase;

public:
    using ContentDescrVector = std::vector<ContentDescr>;

public:
    /** \addtogroup getElementType
	 *  @
	 {
	 */

    //========================================
    //! \brief Return the ElementType of the type of the
    //!        given parameter \a t.
    //! \tparam    T  Type that can be handled by the
    //!               variable contents mechanism of
    //!               DeviceStatus6303, i.e. for which
    //!               an ElementType is defined.
    //! \param[in] t  Any variable of type T.
    //! \return ElementType corresponding to the type T.
    //----------------------------------------
    template<typename T>
    static inline ElementType getElementType(const T& t);
    /** @}*/

    //========================================
    //! \brief Determine the number of elements from given
    //!        ElementType and total size.
    //! \param[in] et         ElementType of the elements.
    //! \param[in] nbOfBytes  Total size in bytes.
    //! \return The number of elements of type ElementTypes
    //!         fit into \a nbOfBytes bytes.
    //----------------------------------------
    static uint8_t getNbOfElements(const ElementType et, const uint8_t nbOfBytes);

public:
    DeviceStatus6303() : SpecializedDataContainer() {}
    DeviceStatus6303(const DeviceStatus6303&) = delete;
    DeviceStatus6303& operator=(const DeviceStatus6303&) = delete;
    virtual ~DeviceStatus6303()                          = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter for the fixed part of the DeviceStatus6303 data type
        //========================================
        //! \brief Get the serial number of the scanner.
        //! \return The serial number of the scanner.
        //----------------------------------------
    const SerialNumberIn6303& getSerialNumberOfScanner() const { return this->serialNumberOfScanner; }
    //	SerialNumberIn6303& getSerialNumberOfScanner() { return this->serialNumberOfScanner; }

    //========================================
    //! \brief Get the type of the scanner.
    //! \return The type of the scanner.
    //----------------------------------------
    uint8_t getScannerType() const { return scannerType; }

    //========================================
    //! \brief Get the device id of the scanner.
    //! \return The device id of the scanner.
    //----------------------------------------
    uint8_t getDeviceId() const { return deviceId; }

    //========================================
    //! \brief Get the FPGA version of the scanner.
    //! \return The FPGA version of the scanner.
    //----------------------------------------
    const Version448In6303& getFpgaVersion() const { return this->fpgaVersion; }
    //	Version448In6303& getFpgaVersion() { return this->fpgaVersion; }

    //========================================
    //! \brief Get the host version of the scanner.
    //! \return The host version of the scanner.
    //----------------------------------------
    const Version448In6303& getHostVersion() const { return this->hostVersion; }
    //	Version448In6303& getHostVersion() { return this->hostVersion; }

    //========================================
    //! \brief Get the FPGA status register.
    //! \return The FPGA status register.
    //----------------------------------------
    uint16_t getFpgaStatusRegister() const { return this->fpgaStatusRegister; }

    //========================================
    //! \brief Get the FPGA operation register.
    //! \return The FPGA operation register.
    //----------------------------------------
    uint16_t getFpgaOperationRegister() const { return this->fpgaOperationRegister; }

    //========================================
    //! \brief Get the scan frequency of the scanner in Hz.
    //!
    //! This method has been introduced for convenience.
    //! The DeviceStatus6303 naturally will provide the
    //! reciprocal of the scan frequency, the scanPeriod.
    //! \return The scan frequency.
    //! \sa getScanPeriod
    //----------------------------------------
    float getFrequency() const { return 1e6F / float(scanPeriod); }

    //========================================
    //! \brief Get the scan period of the scanner
    //!        in microseconds.
    //! \return The scan period of the scanner.
    //! \sa getFrequency
    //----------------------------------------
    uint32_t getScanPeriod() const { return scanPeriod; }

    //========================================
    //! \brief Get the sensor APD temperature 0.
    //! \return The sensor APD temperature 0.
    //----------------------------------------
    float getSensorTemperatureApd0() const { return sensorTemperatureApd0; }

    //========================================
    //! \brief Get the sensor APD temperature 1.
    //! \return The sensor APD temperature 1.
    //----------------------------------------
    float getSensorTemperatureApd1() const { return sensorTemperatureApd1; }

    //========================================
    //! \brief Get the minimal APD voltage offset.
    //! \return The minimal APD voltage offset.
    //----------------------------------------
    float getMinApdVoltageOffset() const { return minApdVoltageOffset; }

    //========================================
    //! \brief Get the maximal APD voltage offset.
    //! \return The maximal APD voltage offset.
    //----------------------------------------
    float getMaxApdVoltageOffset() const { return maxApdVoltageOffset; }

    //========================================
    //! \brief Get the noise measurement threshold.
    //! \return The noise measurement threshold.
    //----------------------------------------
    uint32_t getNoiseMeasurementThreshold() const { return noiseMeasurementThreshold; }

    //========================================
    //! \brief Get the reference noise.
    //! \return The reference noise.
    //----------------------------------------
    uint16_t getReferenceNoise() const { return referenceNoise; }

public: // setter for the fixed part of the DeviceStatus6303 data type
        //========================================
        //! \brief Set a new serial number.
        //! \param[in] newSerialNumber  New serial number to be set.
        //----------------------------------------
    void setSerialNumber(const SerialNumberIn6303 newSerialNumber) { this->serialNumberOfScanner = newSerialNumber; }

    //========================================
    //! \brief Set a new scanner type.
    //! \param[in] newScannerType  New scanner type to be set.
    //----------------------------------------
    void setScannerType(const uint8_t newScannerType) { this->scannerType = newScannerType; }

    //========================================
    //! \brief Set new device id.
    //! \param[in] newDeviceId  New device id to be set.
    //----------------------------------------
    void setDeviceId(const uint8_t newDeviceId) { this->deviceId = newDeviceId; }

    //========================================
    //! \brief Set new FPGA version.
    //! \param[in] newVersion  New FPGA version to be set.
    //----------------------------------------
    void setFpgaVersion(const Version448In6303& newVersion) { this->fpgaVersion = newVersion; }

    //========================================
    //! \brief Set new host version.
    //! \param[in] newVersion  New host version to be set.
    //----------------------------------------
    void setHostVersion(const Version448In6303& newVersion) { this->hostVersion = newVersion; }

    //========================================
    //! \brief Set new scan period.
    //! \param[in] newScanPeriod  New scan period to be set.
    //----------------------------------------
    void setScanPeriod(const uint32_t newScanPeriod) { scanPeriod = newScanPeriod; }

    //========================================
    //! \brief Set new sensor APD temperature 0.
    //! \param[in] newSensorTemperatureApd0  New sensor APD
    //!                                      temperature 0 to be set.
    //----------------------------------------
    void setSensorTemperatureApd0(const float newSensorTemperatureApd0)
    {
        sensorTemperatureApd0 = newSensorTemperatureApd0;
    }

    //========================================
    //! \brief Set new sensor APD temperature 1.
    //! \param[in] newSensorTemperatureApd1  New sensor APD
    //!                                      temperature 1 to be set.
    //----------------------------------------
    void setSensorTemperatureApd1(const float newSensorTemperatureApd1)
    {
        sensorTemperatureApd1 = newSensorTemperatureApd1;
    }

    //========================================
    //! \brief Set new minimal APD voltage offset.
    //! \param[in] newMinApdVoltageOffset  New minimal APD voltage
    //!                                    offset to be set.
    //----------------------------------------
    void setMinApdVoltageOffset(const float newMinApdVoltageOffset) { minApdVoltageOffset = newMinApdVoltageOffset; }

    //========================================
    //! \brief Set new maximal APD voltage offset.
    //! \param[in] newMaxApdVoltageOffset  New maximal APD voltage
    //!                                    offset to be set.
    //----------------------------------------
    void setMaxApdVoltageOffset(const float newMaxApdVoltageOffset) { maxApdVoltageOffset = newMaxApdVoltageOffset; }

    //========================================
    //! \brief Set new noise measurement threshold.
    //! \param[in] newNoiseMeasurementThreshold
    //!    New noise measurement threshold to be set.
    //----------------------------------------
    void setNoiseMeasurementThreshold(const uint32_t newNoiseMeasurementThreshold)
    {
        noiseMeasurementThreshold = newNoiseMeasurementThreshold;
    }

    //========================================
    //! \brief Set new reference noise.
    //! \param[in] newReferenceNoise  New reference noise to be set.
    //----------------------------------------
    void setReferenceNoise(const uint16_t newReferenceNoise) { referenceNoise = newReferenceNoise; }

public: // methods to handle the variable contents
    bool addContent(const ContentDescr& cd);

    //========================================
    //! \brief Add a (variable) content to the DeviceStatus6303.
    //! \param[in] cId        Id of the content to be added.
    //! \param[in] et         Type of the elements of the
    //!                       content to be added.
    //! \param[in] nbOfBytes  Size of the content in bytes.
    //! \param[in] alignment  Needed Alignment for the data
    //!                       of this content.
    //! \param[in] cData      Pointer to the content data
    //!                       to be added to the DeviceStatus6303.
    //!
    //! \return \c True if the content was added successfully.
    //!         \c false if it failed. Reasons for failure are:
    //!         # The number of content entries has reached
    //!           maxNbOfContentEntries already.
    //!         # The given size (\a nbOfByte) is 0 or negative.
    //!         # The size of the contentBuffer would be exceeded
    //!           by adding the content. (contentBufferSize)
    //!         # A content with the same ContentId already has
    //!           been added.
    //----------------------------------------
    bool addContent(const ContentId cId,
                    const ElementType et,
                    const uint8_t nbOfBytes,
                    const uint32_t alignment,
                    const void* cData);

    //========================================
    //! \brief Add a (variable) content of ElementType STRUCT
    //!        to the DeviceStatus6303.
    //!
    //! \param[in] uds        A user defined structure to be added.
    //!
    //! \return \c True if the content was added successfully.
    //!         \c false if it failed. Reasons for failure are:
    //!         # The number of content entries has reached
    //!           maxNbOfContentEntries already.
    //!         # The given size (\a nbOfByte) is 0 or negative.
    //!         # The size of the contentBuffer would be exceeded
    //!           by adding the content. (contentBufferSize)
    //!         # A content with the same ContentId already has
    //!           been added.
    //----------------------------------------
    bool addContent(const UserDefinedStructBase& uds);

    //========================================
    //! \brief Find a content with the given id.
    //! \param[in] cId  Id of the content to be
    //!                 found.
    //! \return On success findContent will return
    //!         the index of the content. If there
    //!         is no such content, -1 will be
    //!         returned.
    //----------------------------------------
    int findContent(const ContentId cId);

    //========================================
    //! \brief Get the data of a (variable) content. User defined
    //!        structs cannot be read by this method.
    //! \tparam     T      Type of the data to be
    //!                    expected.
    //! \param[in]  cId    Id of the content.
    //! \param[out] cData  On exit and on success
    //!                    it will contain the
    //!                    address of the content data.
    //!                    If the operation was not
    //!                    successful \a cData will be
    //!                    nullptr.
    //!
    //! \return On success, the size of the content
    //!         in bytes will be returned. If a
    //!         content with the given id \a cId was
    //!         not found -1 will be returned, if
    //!         the ElementType of the content does
    //!         not match \a T, -2 will be returned.
    //! \remark No user defined structs can be read with
    //!         this method. Use
    //!         getContent(const ContentId, UserDefinedStructBase&)
    //!         instead.
    //----------------------------------------
    template<typename T>
    inline int16_t getContent(const ContentId cId, const T*& cData);

    //========================================
    //! \brief Get the data of a (variable) content of
    //!        contents that contain an user defined
    //!        struct.
    //! \param[in]       cId    Id of the content.
    //! \param[in, out]  uds    A user defined struct to
    //!                         be filled.
    //!
    //! \return \c True in case of success. \c false
    //!         in case the content could not be found,
    //!         the content is not of element type Struct
    //!         or the read method of \a uds failed.
    //----------------------------------------
    bool getContent(const ContentId cId, UserDefinedStructBase& uds);

    //========================================
    //! \brief Return the vector of all content descriptions.
    //!
    //! The vector received by this method can be used to
    //! traverse through all content entries.
    //! \return The vector that contains all content descriptions.
    //----------------------------------------
    const ContentDescrVector& getContentDescrs() const { return m_contentEntries; }

protected:
    void clear()
    {
        m_contentEntries.clear();
        m_usedBytesInContentData = 0;
    }

public:
    //========================================
    //! \brief Maximal number of (variable) contents
    //!        that can be handled by DeviceStatus6303.
    //----------------------------------------
    static const unsigned int maxNbOfContentEntries = 30;

    //========================================
    //! \brief Size of the buffer to hold the data
    //!        of all (variable) contents.
    //!        (Shared buffer)
    //----------------------------------------
    static const unsigned int contentBufferSize = 1024;

protected:
    SerialNumberIn6303 serialNumberOfScanner{}; //!< Serial number of the scanner.
    uint8_t scannerType{0}; //!< Type of the scanner.
    uint8_t deviceId{0}; //!< Device id of the scanner.

    Version448In6303 fpgaVersion{}; //!< Version of the FPGA.
    Version448In6303 hostVersion{}; //!< Version of the host.

    uint16_t fpgaStatusRegister{0}; //!< State of the FPGA status register.
    uint16_t fpgaOperationRegister{0}; //!< State of the FPGA operation register.

    uint32_t scanPeriod{0}; //!< Scan period in usec.

    float sensorTemperatureApd0{0.0F}; //!< Sensor APD temperature 0 in °C.
    float sensorTemperatureApd1{0.0F}; //!< Sensor APD temperature 1 in °C.
    float minApdVoltageOffset{0.0F}; //!< Minimal APD voltage offset.
    float maxApdVoltageOffset{0.0F}; //!< Maximal APD voltage offset.

    uint32_t noiseMeasurementThreshold{0}; //!< Noise measurement threshold.
    uint16_t referenceNoise{0}; //!< Reference noise.

protected:
    //========================================
    //! \brief Static vector to hold the ContentDescr
    //!        for the up to \a maxNbOfContentEntries
    //!        added (variable) contents).
    //----------------------------------------
    ContentDescrVector m_contentEntries{};

    //========================================
    //! \brief Shared buffer to hold the data of
    //!        all added contents.
    //----------------------------------------
    char m_contentData[contentBufferSize];

    //========================================
    //! \brief Number of bytes already used by
    //!        the added contents.
    //!
    //! This value cannot be larger than
    //! \a contentBufferSize.
    //----------------------------------------
    unsigned int m_usedBytesInContentData{0};
}; // DeviceStatus6303

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief An content entry in the variable part
//!        of this data type.
//! \date Jul 30, 2013
//!
//! An object of this class describe a content entry
//! in the variable part of this data type. Apart from
//! the ContentId, the ElementType and the number of
//! used bytes it also contains a pointer to the content
//! which is located in the content buffer of the
//! DeviceStatus6303 object.
//------------------------------------------------------------------------------
class DeviceStatus6303::ContentDescr
{
public:
    //========================================
    //! \brief Default construtcor.
    //----------------------------------------
    ContentDescr() : m_contentId{ContentId::Illegal}, m_elementType{ElementType::Illegal}, m_nbOfBytes{0} {}

    //========================================
    //! \brief Constructor.
    //! \param[in] cId        ContentId of the content.
    //! \param[in] et         ElementType of the content.
    //! \param[in] nbOfBytes  Size of the content in bytes.
    //----------------------------------------
    ContentDescr(const ContentId cId, const ElementType et, const uint8_t nbOfBytes)
      : m_contentId{cId}, m_elementType{et}, m_nbOfBytes{nbOfBytes}
    {}

    virtual ~ContentDescr() {}

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool serialize(std::ostream& os) const;

public:
    //========================================
    //! \brief Get the ContentId of this content.
    //! \return ContentId of this content.
    //----------------------------------------
    ContentId getContentId() const { return m_contentId; }

    //========================================
    //! \brief Get the ElementType of this content.
    //! \return ElementType of this content.
    //----------------------------------------
    ElementType getElementType() const { return m_elementType; }

    //========================================
    //! \brief Get the size of the content in bytes.
    //! \return The size of this content in bytes.
    //----------------------------------------
    uint8_t getNbOfBytes() const { return m_nbOfBytes; }

    //========================================
    //! \brief Get the content buffer pointer.
    //! \return Pointer to the content.
    //----------------------------------------
    const char* getContentBuffer() const { return m_contentBuffer; }

    uint8_t getNeededBufferAlignment() const;

public:
    //========================================
    //! \brief Set the content buffer pointer.
    //! \param[in] buffer  Set the contentBuffer to the
    //!                    \a buffer. \a buffer contains
    //!                    an address inside the
    //!                    DeviceStatus6303's content
    //!                    buffer.
    //----------------------------------------
    void setContentBuffer(char* const buffer) { m_contentBuffer = buffer; }

protected:
    //========================================
    //! \brief Serialized the content's elements
    //!        into the given buffer.
    //! \tparam          T          Type of the elements of
    //!                             the content.
    //! \param[in, out]  os         Target stream, where the
    //!                             content's elements will
    //!                             be serialized to.
    //! \param[in]       nbOfBytes  Total size of all elements
    //!                             to be written.
    //! \param[in]       elements   Elements of the content
    //!                             to be written.
    //! \return Number of written bytes.
    //----------------------------------------
    template<typename T>
    static void writeVcElements(std::ostream& os, const uint32_t nbOfBytes, const T* const elements);

protected:
    ContentId m_contentId; //!< ContentId of the content.
    ElementType m_elementType; //!< ElementType of the content
    uint8_t m_nbOfBytes; //!< Size of the content in bytes.

    //========================================
    //! \brief Pointer to the content itself inside
    //!        the DeviceStatus6303 object's content
    //!        buffer.
    //----------------------------------------
    char* m_contentBuffer{nullptr};
}; // DeviceStatus6303::ContentDescr

//==============================================================================

class DeviceStatus6303::ContentDescrDeserializer : public ContentDescr
{
public:
    ContentDescrDeserializer() : ContentDescr() { this->setContentBuffer(deserializeBuffer); }

public:
    bool deserialize(std::istream& is);

protected:
    ContentId readContentId(std::istream& is);
    ElementType readElementType(std::istream& is);

    //========================================
    //! \brief Deserialized the content's elements
    //!        into the given buffer.
    //! \tparam          T                   Type of the elements of
    //!                                      the content.
    //! \param[in]       is                 Source buffer, where the
    //!                                      content's elements will
    //!                                      be deserialized from.
    //! \param[in]       nbOfElementsToWrite  Total size of all elements
    //!                                      to be write.
    //! \param[out]       elements            Target buffer, where the
    //!                                      content elements will be
    //!                                      read into.
    //! \return Number of read bytes.
    //----------------------------------------
    template<typename T>
    static void readVcElements(std::istream& is, const int nbOfElementsToWrite, T* const elements);

protected:
    static const int maxSizeOfContent = 255;
    static char deserializeBuffer[maxSizeOfContent + 1];
}; // DeviceStatus6303::ContentDescrDeserializer

//==============================================================================
//!\brief  Base class for user define structures.
//!\remark Derived class have to have a default constructor.
//------------------------------------------------------------------------------
class DeviceStatus6303::UserDefinedStructBase
{
public:
    UserDefinedStructBase(const ContentId contentId);
    virtual ~UserDefinedStructBase();

public:
    virtual bool deserialize(const ContentDescr& cd) = 0;
    virtual bool serialize(char*& buf) const         = 0;
    virtual uint8_t getSerializedSize() const        = 0;

public:
    virtual ContentId getContentId() const { return m_contentId; }
    virtual ElementType getElementType() const { return m_elementType; }

protected:
    const ContentId m_contentId;
    const ElementType m_elementType{ElementType::STRUCT};
}; // DeviceStatus6303::UserDefinedStructBase

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================

/** \addtogroup getElementType
 *  @
 {
 */

//==============================================================================
//! \brief Get the #ElementType of type uint8_t.
//!
//! This method is an \c uint8_t specialization of DeviceStatus6303::getElementType<T>.
//!
//! \return #UINT8, the ElementType of type uint8_t.
//! \sa DeviceStatus6303::getElementType<T>.
//------------------------------------------------------------------------------
template<>
inline DeviceStatus6303::ElementType DeviceStatus6303::getElementType<uint8_t>(const uint8_t&)
{
    return ElementType::UINT8;
}

//==============================================================================
//! \brief Get the #ElementType of type int8_t.
//!
//! This method is an \c int8_t specialization of DeviceStatus6303::getElementType<T>.
//!
//! \return #INT8, the ElementType of type int8_t.
//! \sa DeviceStatus6303::getElementType<T>.
//------------------------------------------------------------------------------
template<>
inline DeviceStatus6303::ElementType DeviceStatus6303::getElementType<int8_t>(const int8_t&)
{
    return ElementType::INT8;
}

//==============================================================================
//! \brief Get the #ElementType of type uint16_t.
//!
//! This method is an \c uint16_t specialization of DeviceStatus6303::getElementType<T>.
//!
//! \return #UINT16, the ElementType of type uint16_t.
//! \sa DeviceStatus6303::getElementType<T>.
//------------------------------------------------------------------------------
template<>
inline DeviceStatus6303::ElementType DeviceStatus6303::getElementType<uint16_t>(const uint16_t&)
{
    return ElementType::UINT16;
}

//==============================================================================
//! \brief Get the #ElementType of type int16_t.
//!
//! This method is an \c int16_t specialization of DeviceStatus6303::getElementType<T>.
//!
//! \return #INT16, the ElementType of type int16_t.
//! \sa DeviceStatus6303::getElementType<T>.
//------------------------------------------------------------------------------
template<>
inline DeviceStatus6303::ElementType DeviceStatus6303::getElementType<int16_t>(const int16_t&)
{
    return ElementType::INT16;
}

//==============================================================================
//! \brief Get the #ElementType of type uint32_t.
//!
//! This method is an \c uint32_t specialization of DeviceStatus6303::getElementType<T>.
//!
//! \return #UINT32, the ElementType of type uint32_t.
//! \sa DeviceStatus6303::getElementType<T>.
//------------------------------------------------------------------------------
template<>
inline DeviceStatus6303::ElementType DeviceStatus6303::getElementType<uint32_t>(const uint32_t&)
{
    return ElementType::UINT32;
}

//==============================================================================
//! \brief Get the #ElementType of type int32_t.
//!
//! This method is an \c int32_t specialization of DeviceStatus6303::getElementType<T>.
//!
//! \return #INT32, the ElementType of type int32_t.
//! \sa DeviceStatus6303::getElementType<T>.
//------------------------------------------------------------------------------
template<>
inline DeviceStatus6303::ElementType DeviceStatus6303::getElementType<int32_t>(const int32_t&)
{
    return ElementType::INT32;
}

//==============================================================================
//! \brief Get the #ElementType of type float.
//!
//! This method is an \c float specialization of DeviceStatus6303::getElementType<T>.
//!
//! \return #FLOAT32, the ElementType of type float.
//! \sa DeviceStatus6303::getElementType<T>.
//------------------------------------------------------------------------------
template<>
inline DeviceStatus6303::ElementType DeviceStatus6303::getElementType<float>(const float&)
{
    return ElementType::FLOAT32;
}

//==============================================================================
//! \brief Return the ElementType of the type of the
//!        given parameter \a t.
//! \tparam    T  Type that can be handled by the
//!               variable contents mechanism of
//!               DeviceStatus6303, i.e. for which
//!               an ElementType is defined.
//! \param[in] t  Any variable of type T.
//!
//! Since all other case covered by template specializations
//! of DeviceStatus6303::getElementType<T> the only element type
//! left is #STRUCT.
//!
//! \return #STRUCT.
//! \sa DeviceStatus6303::getElementType<uint8_t>, DeviceStatus6303::getElementType<int8_t>,
//!     DeviceStatus6303::getElementType<uint16_t>, DeviceStatus6303::getElementType<int16_t>,
//!     DeviceStatus6303::getElementType<uint32_t>, DeviceStatus6303::getElementType<int32_t>,
//!     DeviceStatus6303::getElementType<float>
//------------------------------------------------------------------------------
template<typename T>
DeviceStatus6303::ElementType DeviceStatus6303::getElementType(const T&)
{
    return ElementType::STRUCT;
}

/** @}*/

//==============================================================================

template<typename T>
inline int16_t DeviceStatus6303::getContent(const ContentId cId, const T*& cData)
{
    const int32_t idx = findContent(cId);
    if (idx == -1)
    {
        cData = nullptr;
        return -1;
    }

    const ContentDescr& c = m_contentEntries.at(static_cast<uint32_t>(idx));
    if (c.getElementType() == ElementType::STRUCT)
    {
        return -2;
    }

    if (DeviceStatus6303::getElementType<T>(*cData) != c.getElementType())
    {
        cData = nullptr;
        return -3;
    }

    cData = reinterpret_cast<const T*>(c.getContentBuffer());
    return c.getNbOfBytes();
}

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Comparison operator for equality.
//!
//! Compare this object with another DeviceStatus6303
//! object \a rhs.
//!
//! \param[in] rhs  Other DeviceStatus6303
//!                 this object will be compared
//!                 with.
//! \return \c True if the content of this object
//!         is equal to the content of the other
//!         object \a rhs. \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const DeviceStatus6303& lhs, const DeviceStatus6303& rhs);

inline bool operator!=(const DeviceStatus6303& lhs, const DeviceStatus6303& rhs) { return !(lhs == rhs); }

//==============================================================================

std::ostream& operator<<(std::ostream& os, const DeviceStatus6303::ContentId cId);
std::ostream& operator<<(std::ostream& os, const DeviceStatus6303::ElementType et);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
