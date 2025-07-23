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
//! \brief Data type of iutp message which define the binary format of payload.
//------------------------------------------------------------------------------
enum class IutpDataType : uint16_t
{
    Unknown                   = 0x00U, //!< Unkown data type.
    GenericDatatypeProtocol   = 0x01U, //!< deprecated, Hermes bridge generic datatype with IcdHeader in little endian.
    GenericDatatypeProtocolBE = 0x02U, //!< Hermes bridge generic datatype with IcdHeader in big endian.
    IntpData                  = 0x11U //!< Intp data wrapped.
};

//==============================================================================
//! \brief Data package to store iutp messages.
//!
//! The IUTP is a small and scalable transport protocol on top of UDP.
//! In its current version it allows to detect messages received out of order and missing fragments
//! The IUTP is not reliable, thus it cannot assure a transmission was successfully received.
//!
//! The protocol allows to transfer messages of a maximum payload size of approximately 4GB.
//------------------------------------------------------------------------------
class IutpNetworkDataPackage : public NetworkDataPackage
{
public:
    //========================================
    //! \brief Construct iutp message with required header.
    //! \param[in] streamId         Id of the data stream.
    //! \param[in] sequenceNumber   Message index of stream.
    //! \param[in] dataType         Datatype id of binary format.
    //! \param[in] index            Index of the package over all streams.
    //----------------------------------------
    IutpNetworkDataPackage(const uint8_t streamId,
                           const uint16_t sequenceNumber,
                           const IutpDataType dataType,
                           const int64_t index)
      : NetworkDataPackage{index}, m_streamId{streamId}, m_sequenceNumber{sequenceNumber}, m_dataType{dataType}
    {}

    //========================================
    //! \brief Construct iutp message with required header.
    //! \param[in] streamId         Id of the data stream.
    //! \param[in] sequenceNumber   Message index of stream.
    //! \param[in] dataType         Datatype id of binary format.
    //! \param[in] index            Index of the package over all streams.
    //! \param[in] sourceUri        Source Uri.
    //! \param[in] destinationUri   Destination Uri.
    //----------------------------------------
    IutpNetworkDataPackage(const uint8_t streamId,
                           const uint16_t sequenceNumber,
                           const IutpDataType dataType,
                           const int64_t index,
                           const Uri& sourceUri,
                           const Uri& destinationUri)
      : NetworkDataPackage{index, sourceUri, destinationUri},
        m_streamId{streamId},
        m_sequenceNumber{sequenceNumber},
        m_dataType{dataType}
    {}

    //========================================
    //! \brief Construct iutp message with required header.
    //! \param[in] streamId         Id of the data stream.
    //! \param[in] sequenceNumber   Message index of stream.
    //! \param[in] dataType         Datatype id of binary format.
    //! \param[in] index            Index of the package over all streams.
    //! \param[in] sourceUri        Source Uri.
    //! \param[in] destinationUri   Destination Uri.
    //! \param[in] payload          Data payload.
    //----------------------------------------
    IutpNetworkDataPackage(const uint8_t streamId,
                           const uint16_t sequenceNumber,
                           const IutpDataType dataType,
                           const int64_t index,
                           const Uri& sourceUri,
                           const Uri& destinationUri,
                           const PayloadType& payload)
      : NetworkDataPackage{index, sourceUri, destinationUri, payload},
        m_streamId{streamId},
        m_sequenceNumber{sequenceNumber},
        m_dataType{dataType}
    {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IutpNetworkDataPackage() override = default;

public:
    //========================================
    //! \brief Compare data packages for equality.
    //! \param[in] lhs  IutpNetworkDataPackage to compare.
    //! \param[in] rhs  IutpNetworkDataPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //! \note This compares aside the payload the stream id, sequence number and data type.
    //----------------------------------------
    friend bool operator==(const IutpNetworkDataPackage& lhs, const IutpNetworkDataPackage& rhs)
    {
        return (lhs.getStreamId() == rhs.getStreamId()) || (lhs.getSequenceNumber() == rhs.getSequenceNumber())
               || (lhs.getDataType() == rhs.getDataType()) || (lhs.getPayload() == rhs.getPayload());
    }

    //========================================
    //! \brief Compare data packages for inequality.
    //! \param[in] lhs  IutpNetworkDataPackage to compare.
    //! \param[in] rhs  IutpNetworkDataPackage to compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //! \note This compares aside the payload the stream id, sequence number and data type.
    //----------------------------------------
    friend bool operator!=(const IutpNetworkDataPackage& lhs, const IutpNetworkDataPackage& rhs)
    {
        return !(lhs == rhs);
    }

public: // getter
    //========================================
    //! \brief Get streamId of iutp message.
    //! \returns StreamId of iutp message.
    //----------------------------------------
    uint8_t getStreamId() const { return this->m_streamId; }

    //========================================
    //! \brief Get sequenceNumber of iutp message.
    //! \returns SequenceNumber of iutp message.
    //----------------------------------------
    uint16_t getSequenceNumber() const { return this->m_sequenceNumber; }

    //========================================
    //! \brief Get dataType of iutp message.
    //! \returns DataType of iutp message.
    //----------------------------------------
    IutpDataType getDataType() const { return this->m_dataType; }

public: // setter
    //========================================
    //! \brief Set streamId of iutp message.
    //! \param[in] streamId  New streamId.
    //----------------------------------------
    void setStreamId(const uint8_t streamId) { this->m_streamId = streamId; }

    //========================================
    //! \brief Set sequenceNumber of iutp message.
    //! \param[in] sequenceNumber  New sequenceNumber.
    //----------------------------------------
    void setSequenceNumber(const uint16_t seqNumber) { this->m_sequenceNumber = seqNumber; }

    //========================================
    //! \brief Set dataType of iutp message.
    //! \param[in] dataType  New dataType.
    //----------------------------------------
    void setDataType(const IutpDataType dataType) { this->m_dataType = dataType; }

private:
    //========================================
    //! \brief Stream identifier of the IUTP message stream in which the message was transferred.
    //----------------------------------------
    uint8_t m_streamId;

    //========================================
    //! \brief Sequence number unique for each tuple of <endpoint, stream id>.
    //----------------------------------------
    uint16_t m_sequenceNumber;

    //========================================
    //! \brief Datatype of the payload of the message as taken from initial fragment.
    //----------------------------------------
    IutpDataType m_dataType;
};

//==============================================================================

//========================================
//! \brief Nullable IutpNetworkDataPackage pointer.
//----------------------------------------
using IutpNetworkDataPackagePtr = std::shared_ptr<IutpNetworkDataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
