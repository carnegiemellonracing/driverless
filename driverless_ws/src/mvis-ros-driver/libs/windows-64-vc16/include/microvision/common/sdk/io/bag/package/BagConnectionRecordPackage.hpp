//==============================================================================
//! \file
//!
//! \brief Data package to store BAG connection record.
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
//! \brief Data package to store BAG connection record.
//! \extends BagRecordPackage
//------------------------------------------------------------------------------
class BagConnectionRecordPackage : public BagRecordPackage
{
public:
    //========================================
    //! \brief BAG record header field name for connection.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameConnection;

    //========================================
    //! \brief BAG record header field name for topic.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string recordHeaderFieldNameTopic;

    //========================================
    //! \brief BAG connection header field name for topic.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string connectionHeaderFieldNameTopic;

    //========================================
    //! \brief BAG connection header field name for type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string connectionHeaderFieldNameType;

    //========================================
    //! \brief BAG connection header field name for message definition.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string connectionHeaderFieldNameMessageDefinition;

    //========================================
    //! \brief BAG connection header field name for type md5 checksum.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string connectionHeaderFieldNameMessageDefinitionChecksum;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagConnectionRecordPackage";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Construct BAG connection record with sequence index and source URI.
    //! \param[in] index  Position in the data package stream, depends to source.
    //! \param[in] uri    Source of data (stream) as Uri.
    //----------------------------------------
    BagConnectionRecordPackage(const int64_t index, const Uri& sourceUri);

    //========================================
    //! \brief Construct BAG connection record with all posible parameters.
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source of data (stream) as Uri.
    //! \param[in] payload  BAG record data.
    //! \param[in] version  BAG format version.
    //! \param[in] header   BAG record header.
    //----------------------------------------
    BagConnectionRecordPackage(const int64_t index,
                               const Uri& sourceUri,
                               const SharedBuffer& payload,
                               const BagFormatVersion version,
                               const BagRecordFieldMap& header);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagConnectionRecordPackage() override;

public: // getter
    //========================================
    //! \brief Unique connection ID.
    //! \details The header key is 'conn' (connection).
    //! \returns The id of the connection.
    //----------------------------------------
    uint32_t getId() const;

    //========================================
    //! \brief Topic on which the messages are stored.
    //! \details The header key is 'topic' (topic).
    //! \returns Topic of data.
    //----------------------------------------
    std::string getTopic() const;

public: // setter
    //========================================
    //! \brief Unique connection ID.
    //! \details The header key is 'conn' (connection).
    //! \param[in] value  The id of the connection.
    //----------------------------------------
    void setId(const uint32_t value);

    //========================================
    //! \brief Topic on which the messages are stored.
    //! \details The header key is 'topic' (topic).
    //! \param[in] value  Topic of data.
    //----------------------------------------
    void setTopic(const std::string& value);

public:
    //========================================
    //! \brief BAG record has valid header fields.
    //! \return Either \c true if header has all guaranteed fields, otherwise \c false.
    //----------------------------------------
    bool isValid() const override;

    //========================================
    //! \brief Read connection header with data type etc.
    //! \param[out] header  Connection header fields.
    //! \return Either \c true if connection header successful readed, otherwise \c false.
    //----------------------------------------
    bool readConnectionHeader(BagRecordFieldMap& header);

    //========================================
    //! \brief Write connection header with data type etc.
    //! \param[in] header  Connection header fields.
    //! \return Either \c true if connection header successful written, otherwise \c false.
    //----------------------------------------
    bool writeConnectionHeader(const BagRecordFieldMap& header);

}; // class BagConnectionRecordPackage

//==============================================================================

//==============================================================================
//! \brief Nullable BagConnectionRecordPackage pointer.
//------------------------------------------------------------------------------
using BagConnectionRecordPackagePtr = std::shared_ptr<BagConnectionRecordPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
