//==============================================================================
//! \file
//!
//! \brief Defines the BAG data package for serialized data in BAG stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 11, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>

#include <microvision/common/sdk/io/DataPackage.hpp>
#include <microvision/common/sdk/Ptp64Time.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Meta information of BAG data package to map data by topic and data type.
//------------------------------------------------------------------------------
class BagDataHeader final
{
public:
    //========================================
    //! \brief Default constructor.
    //! \note All fields except id are required so please set them.
    //----------------------------------------
    BagDataHeader();

    //========================================
    //! \brief Construct header in case of unknown message definition.
    //! \note All fields except id are required so please set them.
    //! \param[in] id        Unique connection id.
    //! \param[in] topic     Subscribed topic.
    //! \param[in] dataType  Data type of serilized payload.
    //----------------------------------------
    BagDataHeader(const uint32_t id, const std::string& topic, const std::string& dataType);

    //========================================
    //! \brief Construct header with required information.
    //! \param[in] id                            Unique connection id.
    //! \param[in] topic                         Subscribed topic.
    //! \param[in] dataType                      Data type of serilized payload.
    //! \param[in] messageDefinition             Textual definition of data type.
    //! \param[in] messageDefinitionMd5Checksum  Md5 checksum of cleaned message definition.
    //----------------------------------------
    BagDataHeader(const uint32_t id,
                  const std::string& topic,
                  const std::string& dataType,
                  const std::string& messageDefinition,
                  const std::string& messageDefinitionMd5Checksum);

public:
    //========================================
    //! \brief Compare two bag data header for equality.
    //! \param[in] lhs  BagDataHeader to compare.
    //! \param[in] rhs  BagDataHeader to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const BagDataHeader& lhs, const BagDataHeader& rhs);

    //========================================
    //! \brief Compare two bag data header for inequality.
    //! \param[in] lhs  BagDataHeader to compare.
    //! \param[in] rhs  BagDataHeader to compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const BagDataHeader& lhs, const BagDataHeader& rhs);

public: // getter
    //========================================
    //! \brief Get connection id which is unique over all headers in the same BAG stream.
    //! \returns Unique connection id.
    //----------------------------------------
    uint32_t getId() const;

    //========================================
    //! \brief Get topic on which the data subscribed.
    //! \returns Subscribed topic.
    //----------------------------------------
    const std::string& getTopic() const;

    //========================================
    //! \brief Get data type in which format the payload did serialized.
    //! \note For version check of data type use message definition md5 checksum.
    //! \returns Data type of serilized payload.
    //----------------------------------------
    const std::string& getDataType() const;

    //========================================
    //! \brief Get textual definition of data type.
    //! \note See http://wiki.ros.org/msg for more information about message definition.
    //! \returns Message definition.
    //----------------------------------------
    const std::string& getMessageDefinition() const;

    //========================================
    //! \brief Get md5 checksum of cleaned message definition.
    //! \note See http://wiki.ros.org/ROS/Technical%20Overview#Message_serialization_and_msg_MD5_sums
    //!       for more information about cleaning message definition.
    //! \returns Md5 checksum.
    //----------------------------------------
    const std::string& getMessageDefinitionMd5Checksum() const;

public: // setter
    //========================================
    //! \brief Set connection id which is unique in BAG stream.
    //! \param[in] id  Unique connection id.
    //----------------------------------------
    void setId(const uint32_t id);

    //========================================
    //! \brief Set topic on which the data subscribed.
    //! \param[in] topic  Subscribed topic.
    //----------------------------------------
    void setTopic(const std::string& topic);

    //========================================
    //! \brief Set data type in which format the payload did serialized.
    //! \note For version check of data type use message definition md5 checksum.
    //! \param[in] dataType  Data type of serilized payload.
    //----------------------------------------
    void setDataType(const std::string& dataType);

    //========================================
    //! \brief Set textual definition of data type.
    //! \note See http://wiki.ros.org/msg for more information about message definition.
    //! \param[in] messageDefinition  Message definition.
    //----------------------------------------
    void setMessageDefinition(const std::string& messageDefinition);

    //========================================
    //! \brief Set md5 checksum of cleaned message definition.
    //! \note See http://wiki.ros.org/ROS/Technical%20Overview#Message_serialization_and_msg_MD5_sums
    //!       for more information about cleaning message definition.
    //! \param[in] messageDefinitionMd5Checksum  Md5 checksum.
    //----------------------------------------
    void setMessageDefinitionMd5Checksum(const std::string& messageDefinitionMd5Checksum);

private:
    //========================================
    //! \brief Unique connection id.
    //----------------------------------------
    uint32_t m_id;

    //========================================
    //! \brief Subscribed topic.
    //----------------------------------------
    std::string m_topic;

    //========================================
    //! \brief Data type of serilized payload.
    //----------------------------------------
    std::string m_dataType;

    //========================================
    //! \brief Textual definition of data type.
    //! \note See http://wiki.ros.org/msg for more information about message definition.
    //----------------------------------------
    std::string m_messageDefinition;

    //========================================
    //! \brief Md5 checksum of cleaned message definition.
    //! \note See http://wiki.ros.org/ROS/Technical%20Overview#Message_serialization_and_msg_MD5_sums
    //!       for more information about cleaning message definition.
    //----------------------------------------
    std::string m_messageDefinitionMd5Checksum;

}; // class BagDataHeader

//==============================================================================
//! \brief Nullable BagDataHeader pointer.
//------------------------------------------------------------------------------
using BagDataHeaderPtr = std::shared_ptr<BagDataHeader>;

//==============================================================================
//! \brief Package of serialized data and related meta information in BAG stream.
//!
//! \note Payload has to contain 4 byte length (LE) followed by data.
//------------------------------------------------------------------------------
class BagDataPackage final : public DataPackage
{
public:
    //========================================
    //! \brief Construct BAG data package without payload.
    //! \param[in] index        Position in the data package stream, depends to source.
    //! \param[in] sourceUri    Source of data (stream) as Uri.
    //! \param[in] header       Meta information of data. Use new header for every data package.
    //! \param[in] time         Time when the data recorded.
    //----------------------------------------
    BagDataPackage(const int64_t index, const Uri& sourceUri, const BagDataHeaderPtr& header, const Ptp64Time& time);

    //========================================
    //! \brief Construct BAG data package with required information.
    //! \param[in] index        Position in the data package stream, depends to source.
    //! \param[in] sourceUri    Source of data (stream) as Uri.
    //! \param[in] header       Meta information of data. Use new header for every data package.
    //! \param[in] time         Time when the data recorded.
    //! \param[in] payload      Serialized data. Has to contain 4 byte length (LE) followed by data.
    //----------------------------------------
    BagDataPackage(const int64_t index,
                   const Uri& sourceUri,
                   const BagDataHeaderPtr& header,
                   const Ptp64Time& time,
                   const PayloadType& payload);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagDataPackage() override;

public:
    //========================================
    //! \brief Compare two bag data packages for equality.
    //! \param[in] lhs  BagDataPackage to compare.
    //! \param[in] rhs  BagDataPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const BagDataPackage& lhs, const BagDataPackage& rhs);

    //========================================
    //! \brief Compare two bag data packages for inequality.
    //! \param[in] lhs  BagDataPackage to compare.
    //! \param[in] rhs  BagDataPackage to compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const BagDataPackage& lhs, const BagDataPackage& rhs);

public: // getter
    //========================================
    //! \brief Get meta information of serialized data.
    //! \returns Meta information of serialized data.
    //----------------------------------------
    const BagDataHeaderPtr& getHeader() const;

    //========================================
    //! \brief Get time when data recorded.
    //! \returns Time when data recorded.
    //----------------------------------------
    const Ptp64Time& getTime() const;

public: // setter
    //========================================
    //! \brief Set meta information of serialized data.
    //! \param[in] header  Meta information of serialized data.
    //----------------------------------------
    void setHeader(const BagDataHeaderPtr& header);

    //========================================
    //! \brief Get time when data recorded.
    //! \param[in] time  Time when data recorded.
    //----------------------------------------
    void setTime(const Ptp64Time& time);

private:
    //========================================
    //! \brief Meta information of serialized data.
    //----------------------------------------
    BagDataHeaderPtr m_header;

    //========================================
    //! \brief Time when data recorded.
    //----------------------------------------
    Ptp64Time m_time;

}; // class BagDataPackage

//==============================================================================

//==============================================================================
//! \brief Nullable BagDataPackage pointer.
//------------------------------------------------------------------------------
using BagDataPackagePtr = std::shared_ptr<BagDataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
