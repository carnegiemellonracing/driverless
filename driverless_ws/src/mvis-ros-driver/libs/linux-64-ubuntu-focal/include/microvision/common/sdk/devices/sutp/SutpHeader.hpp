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
//! \date Aug 31, 2016
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/logging/logging.hpp>

#include <istream>
#include <string>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Implements the SUTP
//------------------------------------------------------------------------------
class SutpHeader final
{
public:
    //========================================
    //! \brief Size of the serialized header
    //----------------------------------------
    static std::streamsize getSerializedSize_Static() { return 24; }

public:
    //========================================
    //! \brief Sutp Header flags bitfield.
    //----------------------------------------
    enum class SutpFlags : uint8_t
    {
        RawData  = 0x02U, ///< payload is not processed
        TsFormat = 0x04U ///< timestamp format (absolute vs relative)
    };

    //========================================
    //! \brief Enumeration of sutp header
    //!        versions (magic words).
    //----------------------------------------
    enum class ProtocolVersion : uint16_t
    {
        Version_01 = 0x53CAU
    };

public:
    //========================================
    //! \brief Default constructor.
    //!
    //! Fills the header with zeroes. For
    //! details, please refer to the
    //! introduction to class IdcDataHeader.
    //----------------------------------------
    SutpHeader();

    //========================================
    //! \brief Constructor
    //!
    //! For details, please refer to the sutp
    //! specification.
    //----------------------------------------
    SutpHeader(const uint64_t timestamp,
               const uint16_t version,
               const uint16_t seqNo,
               const uint8_t flags,
               const uint8_t scannerId,
               const uint16_t dataType,
               const uint16_t fwVersion,
               const uint16_t scanNo,
               const uint16_t fragsTotal,
               const uint16_t fragNo);

    //========================================
    //! \brief Destructor
    //----------------------------------------
    virtual ~SutpHeader();

public:
    //========================================
    //! \brief Equality predicate
    //----------------------------------------
    bool operator==(const SutpHeader& other) const;

public:
    //========================================
    //! \brief Size of the serialized header
    //----------------------------------------
    std::streamsize getSerializedSize() const { return getSerializedSize_Static(); }

    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    //========================================
    //! \brief get timestamp
    //----------------------------------------
    uint64_t getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief get protocol version
    //----------------------------------------
    uint16_t getVersion() const { return m_version; }

    //========================================
    //! \brief get sequence number
    //----------------------------------------
    uint16_t getSeqNo() const { return m_seqNo; }

    //========================================
    //! \brief get flags
    //----------------------------------------
    uint8_t getFlags() const { return m_flags; }

    //========================================
    //! \brief get scanner id
    //----------------------------------------
    uint8_t getScannerId() const { return m_scannerId; }

    //========================================
    //! \brief get datatype id
    //----------------------------------------
    uint16_t getDatatype() const { return m_dataType; }

    //========================================
    //! \brief get firmware version
    //----------------------------------------
    uint16_t getFwVersion() const { return m_fwVersion; }

    //========================================
    //! \brief get scan number
    //----------------------------------------
    uint16_t getScanNo() const { return m_scanNo; }

    //========================================
    //! \brief get number of total fragments
    //----------------------------------------
    uint16_t getFragsTotal() const { return m_fragsTotal; }

    //========================================
    //! \brief get number of this fragment
    //----------------------------------------
    uint16_t getFragNo() const { return m_fragNo; }

public:
    //========================================
    //! \brief get timestamp
    //----------------------------------------
    void setTimestamp(const uint64_t timestamp) { m_timestamp = timestamp; }

    //========================================
    //! \brief get protocol version
    //----------------------------------------
    void setVersion(const uint16_t version) { m_version = version; }

    //========================================
    //! \brief get sequence number
    //----------------------------------------
    void setSeqNo(const uint16_t seqNo) { m_seqNo = seqNo; }

    //========================================
    //! \brief get flags
    //----------------------------------------
    void setFlags(const uint8_t flags) { m_flags = flags; }

    //========================================
    //! \brief get scanner id
    //----------------------------------------
    void setScannerId(const uint8_t scannerId) { m_scannerId = scannerId; }

    //========================================
    //! \brief get datatype id
    //----------------------------------------
    void setDatatype(const uint16_t datatype) { m_dataType = datatype; }

    //========================================
    //! \brief get firmware version
    //----------------------------------------
    void setFwVersion(const uint16_t fwVersion) { m_fwVersion = fwVersion; }

    //========================================
    //! \brief get scan number
    //----------------------------------------
    void setScanNo(const uint16_t scanNo) { m_scanNo = scanNo; }

    //========================================
    //! \brief get number of total fragments
    //----------------------------------------
    void setFragsTotal(const uint16_t fragsTotal) { m_fragsTotal = fragsTotal; }

    //========================================
    //! \brief get number of this fragment
    //----------------------------------------
    void setFragNo(const uint16_t fragNo) { m_fragNo = fragNo; }

    //========================================
    //! \brief Returns a formatted string of
    //!        header information.
    //----------------------------------------
    std::string prettyPrint() const;

protected:
    static constexpr const char* loggerId = "microvision::common::sdk::SutpHeader";
    static microvision::common::logging::LoggerSPtr logger;

protected:
    //! Timestamp (32 bit seconds, 32 bit nanoseconds)
    uint64_t m_timestamp;

    //! Magic word and protocol version
    uint16_t m_version;

    //! sequence number of this datagram
    uint16_t m_seqNo;

    //========================================
    //! \brief Flags.
    //! \sa SutpFlags
    //----------------------------------------
    uint8_t m_flags;

    //! source of payload
    uint8_t m_scannerId;

    //! DataType of payload
    uint16_t m_dataType;

    //! senders firmware version
    uint16_t m_fwVersion;

    //! corresponding scan id
    uint16_t m_scanNo;

    //! total fragments for complete datatype
    uint16_t m_fragsTotal;

    //! fragment number of this datagram
    uint16_t m_fragNo;
}; // SutpHeader

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
