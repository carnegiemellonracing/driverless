//==============================================================================
//! \file
//!
//! \brief Data package to store BAG message record.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 07, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>

#include <microvision/common/sdk/io/bag/package/BagRecordPackage.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data package to store BAG message record.
//! \extends BagRecordPackage
//------------------------------------------------------------------------------
class BagMessageRecordPackage : public BagRecordPackage
{
public:
    //========================================
    //! \brief BAG record header field name for connection.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameConnection;

    //========================================
    //! \brief BAG record header field name for time.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameTime;

public:
    //========================================
    //! \brief Construct BAG message record with sequence index and source URI.
    //! \param[in] index  Position in the data package stream, depends to source.
    //! \param[in] uri    Source of data (stream) as Uri.
    //----------------------------------------
    BagMessageRecordPackage(const int64_t index, const Uri& sourceUri);

    //========================================
    //! \brief Construct BAG message record with all posible parameters.
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source of data (stream) as Uri.
    //! \param[in] payload  BAG record data.
    //! \param[in] version  BAG format version.
    //! \param[in] header   BAG record header.
    //----------------------------------------
    BagMessageRecordPackage(const int64_t index,
                            const Uri& sourceUri,
                            const SharedBuffer& payload,
                            const BagFormatVersion version,
                            const BagRecordFieldMap& header);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagMessageRecordPackage() override;

public: // getter
    //========================================
    //! \brief ID of the connection where the message arrived.
    //! \details The header key is 'conn' (connection).
    //! \returns The id of the connection.
    //----------------------------------------
    uint32_t getConnectionId() const;

    //========================================
    //! \brief Time at which the message was received.
    //! \details The header key is 'time' (time).
    //! \returns Timestamp when data received.
    //----------------------------------------
    Ptp64Time getTimestamp() const;

public: // setter
    //========================================
    //! \brief ID of connection on which message arrived.
    //! \details The header key is 'conn' (connection).
    //! \param[in] value  The id of the connection.
    //----------------------------------------
    void setConnectionId(const uint32_t value);

    //========================================
    //! \brief Time at which the message was received.
    //! \details The header key is 'time' (time).
    //! \param[in] value  Timestamp when data received.
    //----------------------------------------
    void setTimestamp(const Ptp64Time& value);

public:
    //========================================
    //! \brief BAG record has valid header fields.
    //! \return Either \c true if header has all guaranteed fields, otherwise \c false.
    //----------------------------------------
    bool isValid() const override;

}; // class BagMessageRecordPackage

//==============================================================================

//==============================================================================
//! \brief Nullable BagMessageRecordPackage pointer.
//------------------------------------------------------------------------------
using BagMessageRecordPackagePtr = std::shared_ptr<BagMessageRecordPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
