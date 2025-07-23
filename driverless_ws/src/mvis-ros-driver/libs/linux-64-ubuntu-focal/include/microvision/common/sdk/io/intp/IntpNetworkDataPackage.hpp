//==============================================================================
//!\file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jan 08, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/NetworkDataPackage.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Package type of intp message which defines the binary format of payload.
//------------------------------------------------------------------------------
enum class IntpPackageType : uint16_t
{
    Unknown  = 0x00U, //!< Unknown package type.
    LdmiA200 = 0xB00U, //!< Layered Depth Matrix Interface Static Information.
    LdmiA201 = 0xB01U, //!< Layered Depth Matrix Interface Sensortemperature.
    LdmiA202 = 0xB02U, //!< Layered Depth Matrix Interface SensorIMU.
    LdmiA203 = 0xB03U, //!< Layered Depth Matrix Interface BrokenPixelMessage.
    LdmiA2F0 = 0xBF0U, //!< Layered Depth Matrix Interface Frameheader.
    LdmiA2F1 = 0xBF1U, //!< Layered Depth Matrix Interface RowMessage.
    LdmiA2F2 = 0xBF2U, //!< Layered Depth Matrix Interface Framefooter.
    LdmiA2D0 = 0xBD0U, //!< Static Debug Information.
    LdmiA2D1 = 0xBD1U, //!< Layered Depth Matrix Interface Debug Message.
    LdmiA300 = 0x1B00U //!< Layered Depth Matrix Interface Static Information since B1 sample.
};

//==============================================================================
//! \brief Data package to store intp messages.
//!
//! The INTP message extends the IUTP protocol about checksum of payload,
//! which will be added on send and validated/removed on receive.
//------------------------------------------------------------------------------
class IntpNetworkDataPackage : public NetworkDataPackage
{
public:
    //========================================
    //! \brief Valid intp version used by sensor.
    //----------------------------------------
    static constexpr uint8_t intpVersion{0x2U};

    //========================================
    //! \brief Size of INTP message header.
    //----------------------------------------
    static constexpr std::size_t intpHeaderSize{
        sizeof(uint8_t) + //!< Size of INTP version.
        sizeof(uint16_t) + //!< Size of INTP package type.
        sizeof(uint8_t) //!< Size of sensor id.
    };

    //========================================
    //! \brief Size of INTP message footer.
    //----------------------------------------
    static constexpr std::size_t intpFooterSize{
        sizeof(uint32_t) //!< Size of message checksum.
    };

public:
    //========================================
    //! \brief Construct intp message with required header.
    //! \param[in] packageType      Binary format identifier.
    //! \param[in] sensorNumber     Sensor number to identify who measured the data.
    //! \param[in] index            Index of the package over all streams.
    //----------------------------------------
    IntpNetworkDataPackage(const IntpPackageType packageType, const uint8_t sensorNumber, const int64_t index)
      : NetworkDataPackage{index}, m_version{intpVersion}, m_packageType{packageType}, m_sensorNumber{sensorNumber}
    {}

    //========================================
    //! \brief Construct intp message with required header.
    //! \param[in] packageType      Binary format identifier.
    //! \param[in] sensorNumber     Sensor number to identify who measured the data.
    //! \param[in] index            Index of the package over all streams.
    //! \param[in] sourceUri        Source Uri.
    //! \param[in] destinationUri   Destination Uri.
    //----------------------------------------
    IntpNetworkDataPackage(const IntpPackageType packageType,
                           const uint8_t sensorNumber,
                           const int64_t index,
                           const Uri& sourceUri,
                           const Uri& destinationUri)
      : NetworkDataPackage{index, sourceUri, destinationUri},
        m_version{intpVersion},
        m_packageType{packageType},
        m_sensorNumber{sensorNumber}
    {}

    //========================================
    //! \brief Construct intp message with required header.
    //! \param[in] packageType      Binary format identifier.
    //! \param[in] sensorNumber     Sensor number to identify who measured the data.
    //! \param[in] index            Index of the package over all streams.
    //! \param[in] sourceUri        Source Uri.
    //! \param[in] destinationUri   Destination Uri.
    //! \param[in] payload          Data payload.
    //----------------------------------------
    IntpNetworkDataPackage(const IntpPackageType packageType,
                           const uint8_t sensorNumber,
                           const int64_t index,
                           const Uri& sourceUri,
                           const Uri& destinationUri,
                           const PayloadType& payload)
      : NetworkDataPackage{index, sourceUri, destinationUri, payload},
        m_version{intpVersion},
        m_packageType{packageType},
        m_sensorNumber{sensorNumber}
    {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IntpNetworkDataPackage() override = default;

public:
    //========================================
    //! \brief Compare data packages for equality.
    //! \param[in] lhs  IntpNetworkDataPackage to compare.
    //! \param[in] rhs  IntpNetworkDataPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //! \note This compares aside the payload the package type and sensor number.
    //----------------------------------------
    friend bool operator==(const IntpNetworkDataPackage& lhs, const IntpNetworkDataPackage& rhs)
    {
        return (lhs.getPackageType() == rhs.getPackageType()) || (lhs.getSensorNumber() == rhs.getSensorNumber())
               || (lhs.getPayload() == rhs.getPayload());
    }

    //========================================
    //! \brief Compare data packages for inequality.
    //! \param[in] lhs  IntpNetworkDataPackage to compare.
    //! \param[in] rhs  IntpNetworkDataPackage to compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //! \note This compares aside the payload the package type and sensor number.
    //----------------------------------------
    friend bool operator!=(const IntpNetworkDataPackage& lhs, const IntpNetworkDataPackage& rhs)
    {
        return !(lhs == rhs);
    }

public: // getter
    //========================================
    //! \brief Get version of intp message.
    //! \returns Version of intp message.
    //----------------------------------------
    uint8_t getVersion() const { return this->m_version; }

    //========================================
    //! \brief Get package type of intp message.
    //! \returns Package type of intp message.
    //----------------------------------------
    IntpPackageType getPackageType() const { return this->m_packageType; }

    //========================================
    //! \brief Get sensor number for intp message.
    //! \returns Sensor number for intp message.
    //----------------------------------------
    uint8_t getSensorNumber() const { return this->m_sensorNumber; }

public: // setter
    //========================================
    //! \brief Set version of intp message.
    //! \param[in] version  New version of message.
    //----------------------------------------
    void setVersion(const uint8_t version) { this->m_version = version; }

    //========================================
    //! \brief Set package type of intp message.
    //! \param[in] packageType  New package type of message.
    //----------------------------------------
    void setPackageType(const IntpPackageType packageType) { this->m_packageType = packageType; }

    //========================================
    //! \brief Set sensor number for intp message.
    //! \param[in] sensorNumber  New sensor number.
    //----------------------------------------
    void setSensorNumber(const uint8_t sensorNumber) { this->m_sensorNumber = sensorNumber; }

private:
    //========================================
    //! \brief Get intp message version.
    //----------------------------------------
    uint8_t m_version;

    //========================================
    //! \brief Get package type of payload binary format.
    //----------------------------------------
    IntpPackageType m_packageType;

    //========================================
    //! \brief Sensor number (setup device id for MOVIA - not unique device id) from which the package is sent.
    //----------------------------------------
    uint8_t m_sensorNumber;
}; // class IntpNetworkDataPackage

//==============================================================================
//! \brief Nullable IntpNetworkDataPackage pointer.
//------------------------------------------------------------------------------
using IntpNetworkDataPackagePtr = std::shared_ptr<IntpNetworkDataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
