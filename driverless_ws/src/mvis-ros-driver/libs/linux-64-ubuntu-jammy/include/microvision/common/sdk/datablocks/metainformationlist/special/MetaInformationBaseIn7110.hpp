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
//! \date Apr 5, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/logging/logging.hpp>

#include <microvision/common/sdk/NtpTime.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Snippet containing data for a meta information for datatype 0x7110
//------------------------------------------------------------------------------
class MetaInformationBaseIn7110
{
public:
    //! Maximum length of a string that can be stored as meta information.
    static constexpr uint32_t maxStringLength{std::numeric_limits<uint16_t>::max()};

public:
    //! Type of this meta information bit.
    enum class MetaInformationType : uint16_t
    {
        Unsupported           = 0,
        VersionNumber         = 1,
        AppBaseConfig         = 2,
        AppBaseSyncMethod     = 3,
        EcuId                 = 4,
        Keywords              = 5,
        SyncBoxConfig         = 6,
        ConfigFile            = 7,
        TripMetaData          = 8,
        ProcessingDescription = 9 //!< Meta information type for processing description.
    }; // MetaInformationType

public:
    //========================================
    //! \brief Explicit constructor from type and time.
    //!
    //! \param[in] type       Type of this meta information.
    //! \param[in] timeStamp  Timestamp of this meta information.
    //----------------------------------------
    explicit MetaInformationBaseIn7110(const MetaInformationType type, const NtpTime timeStamp = NtpTime())
      : m_type(type), m_timeStamp(timeStamp)
    {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~MetaInformationBaseIn7110() = default;

public:
    //========================================
    //! \brief Get the type of this meta information.
    //!
    //! \return The type of this meta information.
    //----------------------------------------
    MetaInformationType getType() const { return m_type; }

    //========================================
    //! \brief Set the timestamp of this meta information.
    //!
    //! \param[in] time  The timestamp of this meta information.
    //----------------------------------------
    void setTimestamp(const NtpTime& time) { m_timeStamp = time; }

    //========================================
    //! \brief Get the timestamp of this meta information.
    //!
    //! \return The timestamp of this meta information.
    //----------------------------------------
    const NtpTime& getTimestamp() const { return m_timeStamp; }

public:
    //========================================
    //! \brief Compare this meta information with another one.
    //!
    //! \param[in] other  The other meta information to compare with.
    //! \return \c True, if the two meta information are equal, \c false otherwise.
    //----------------------------------------
    virtual bool isEqual(const MetaInformationBaseIn7110& other) const;

public:
    //========================================
    //! \brief Get the size in bytes that this meta information occupies when being serialized.
    //!
    //! \return The number of bytes used for serialization.
    //----------------------------------------
    virtual std::streamsize getSerializedSize() const;

    //========================================
    //! \brief Read data from the given stream and fill this meta information (deserialization).
    //!
    //! \param[in, out] is      Input data stream
    //! \return \c True if deserialization succeeds, \c false otherwise.
    //----------------------------------------
    virtual bool deserialize(std::istream& is);

    //========================================
    //! \brief Convert this meta information to a serializable format and write it to the given stream (serialization).
    //!
    //! \param[in, out] os      Output data stream
    //! \return \c True if serialization succeeds, \c false otherwise.
    //----------------------------------------
    virtual bool serialize(std::ostream& os) const;

public:
    //========================================
    //! \brief Get the size of the serialized payload.
    //!
    //! \return The size of the serialized payload.
    //----------------------------------------
    virtual uint32_t getSerializedPayloadSize() const = 0;

    //========================================
    //! \brief Deserialize the data from a stream.
    //!
    //! \param[in,out] is           The stream to read the data from.
    //! \param[in]     payloadSize  The size of the payload in the stream.
    //! \return \c True, if the deserialization was successful, \c false otherwise.
    //----------------------------------------
    virtual bool deserializePayload(std::istream& is, const uint32_t payloadSize) = 0;

    //========================================
    //! \brief Serialize the data into a stream.
    //!
    //! \param[in,out] os  The stream to write the data to.
    //! \return \c True, if the serialization was successful, \c false otherwise.
    //----------------------------------------
    virtual bool serializePayload(std::ostream& os) const = 0;

protected:
    //! Type of this meta information container data.
    MetaInformationType m_type{MetaInformationType::Unsupported};
    //! timestamp for this meta information data
    NtpTime m_timeStamp{};

    // logger for all derived meta information types
    static constexpr const char* loggerId = "microvision::common::sdk::MetaInformation";
    static microvision::common::logging::LoggerSPtr logger;
}; // MetaInformationBaseIn7110

//==============================================================================

using MetaInformationBaseIn7110SPtr = std::shared_ptr<MetaInformationBaseIn7110>;

//==============================================================================
//! \brief Test meta information data objects for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator==(const MetaInformationBaseIn7110& lhs, const MetaInformationBaseIn7110& rhs)
{
    return lhs.isEqual(rhs);
}

//==============================================================================
//! \brief Test meta information data objects for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const MetaInformationBaseIn7110& lhs, const MetaInformationBaseIn7110& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
//! \brief Write a textual representation of a meta information type to a stream.
//!
//! \param[in, out] os  The stream to write to.
//! \param[in]      t   The meta information type to write.
//! \return The output stream given in parameter \a os.
//------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const MetaInformationBaseIn7110::MetaInformationType t);

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
