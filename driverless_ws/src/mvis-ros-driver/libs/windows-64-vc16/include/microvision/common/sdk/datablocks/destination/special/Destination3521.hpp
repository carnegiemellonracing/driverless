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
//! \date Jan 25, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief target position as GPS coordinate
//!
//! this data type was generated with the interface to BestMile for fleet management
//! in mind [https://developer.bestmile.com/v1/docs/performing-a-mission#section-mission-fields-overview],
//! it's similar to Destination field
//------------------------------------------------------------------------------
class Destination3521 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static constexpr uint8_t nbOfReserved
        = 4; //!< Number of reserved 32 bit placeholders for adding attributes after initial impl.

public: // type declaration
    using ReservedArray = std::array<uint32_t, nbOfReserved>;

public:
    //==============================================================================
    //! Possible sources of destination data
    //! This type is serialized as UINT8, 255 is the maximum value
    //------------------------------------------------------------------------------
    enum class SourceType : uint8_t
    {
        Mission = 0,
        Unknown = 255
    };

    //==============================================================================
    //! Possible type of destination
    //! This type is serialized as UINT8, 255 is the maximum value
    //------------------------------------------------------------------------------
    enum class DestinationType : uint8_t
    {
        Station          = 0,
        IntermediateHalt = 1,
        Unknown          = 255
    };

    //==============================================================================
    //! Possible purpose types of destination data
    //! This type is serialized as UINT8, 255 is the maximum value
    //------------------------------------------------------------------------------
    enum class PurposeType : uint8_t
    {
        PassengerTransport = 0,
        Maintenance        = 1,
        Unknown            = 255
    };

public:
    //==============================================================================
    //! \brief Constructor with no calls, basically default.
    //------------------------------------------------------------------------------
    Destination3521();

    //==============================================================================
    //! \brief Default destructor.
    //------------------------------------------------------------------------------
    ~Destination3521() override = default;

public:
    constexpr static const char* const containerType{
        "sdk.specialcontainer.destination3521"}; //!< Statically defined type of container.

    //==============================================================================
    //! \brief Get the static hash value of the class id (static version).
    //! \return The hash value specifying the custom data container class.
    //------------------------------------------------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    //==============================================================================
    //! \brief Get the static hash value of the class id. Just returns result of call to \a getClassIdHashStatic().
    //! \return The hash value specifying the custom data container class.
    //------------------------------------------------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // setters and getters
    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective member.
    //! \param[in] targetPosition  Goal position of target in wgs84 format.
    //------------------------------------------------------------------------------
    void setTargetPosition(const PositionWgs84& targetPosition) { this->m_targetPosition = targetPosition; }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective member.
    //! \param[in] targetHeading  Global orientation of target in radian.
    //------------------------------------------------------------------------------
    void setTargetHeading(const double targetHeading) { this->m_targetHeading = targetHeading; }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective member.
    //! \param[in] targetId  Id of target.
    //------------------------------------------------------------------------------
    void setTargetId(const int64_t targetId) { this->m_targetId = targetId; }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective member.
    //! \param[in] timestamp  Time at which this message was generated / sent.
    //------------------------------------------------------------------------------
    void setTimestamp(const NtpTime& timestamp) { this->m_timestamp = timestamp; }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective member.
    //! \param[in] sourceType  Defines source or context this message was generated in. See \a SourceType at top of
    //!                        class for options.
    //------------------------------------------------------------------------------
    void setSourceType(const SourceType sourceType) { this->m_sourceType = sourceType; }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective member.
    //! \param[in] destinationType  Defines the type of destination. See \a DestinationType at top of class for options.
    //------------------------------------------------------------------------------
    void setDestinationType(const DestinationType destinationType) { this->m_destinationType = destinationType; }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective member.
    //! \param[in] purposeType  Defines the purpose in context of missions. See \a PurposeType at top of class for
    //!                         options.
    //------------------------------------------------------------------------------
    void setPurposeType(const PurposeType purposeType) { this->m_purposeType = purposeType; }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective member.
    //! \param[in] id  Id of this message.
    //------------------------------------------------------------------------------
    void setId(const uint32_t id) { this->m_id = id; }

public:
    //==============================================================================
    //! \brief Trivial getter. Returns respective member.
    //! \return Position of the target in wgs84 format.
    //------------------------------------------------------------------------------
    const PositionWgs84& getTargetPosition() const { return m_targetPosition; }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member.
    //! \return Global orientation of the target in radian.
    //------------------------------------------------------------------------------
    double getTargetHeading() const { return m_targetHeading; }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member.
    //! \return Id of the target.
    //------------------------------------------------------------------------------
    int64_t getTargetId() const { return m_targetId; }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member.
    //! \return Time at which the message was generated / sent.
    //------------------------------------------------------------------------------
    const NtpTime& getTimestamp() const { return m_timestamp; }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member.
    //! \return Source or context this message was generated in. See \a SourceType at top of class for options.
    //------------------------------------------------------------------------------
    SourceType getSourceType() const { return m_sourceType; }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member.
    //! \return Type of destination. See \a DestinationType at top of class for options.
    //------------------------------------------------------------------------------
    DestinationType getDestinationType() const { return m_destinationType; }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member.
    //! \return Purpose in context of missions. See \a PurposeType at top of class for options.
    //------------------------------------------------------------------------------
    PurposeType getPurposeType() const { return m_purposeType; }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member.
    //! \return Id of this message.
    //------------------------------------------------------------------------------
    uint32_t getId() const { return m_id; }

    //==============================================================================
    //!\brief Returns value from m_reserved array at specified position \a idx.
    //!\param[in] idx  Index specifying position in \a m_reserved array.
    //!\return Value of \a m_reserved array at position \a idx.
    //!        Returns last element of array, if \a idx >= \a nbOfReserved.
    //------------------------------------------------------------------------------
    uint32_t getReserved(const uint_fast8_t idx) const;

    //==============================================================================
    //!\brief  Trivial getter. Returns respective member.
    //!\return Container of reserved placeholders.
    //------------------------------------------------------------------------------
    const ReservedArray& getReserved() const { return m_reserved; }

protected:
    PositionWgs84 m_targetPosition{}; //!< Wgs84 position of target destination
    double m_targetHeading{0.0}; //!< Global heading/orientation of target in radian.
    int64_t m_targetId{0}; //!< Defines id of the target e.g. parking space id, or edge id
    NtpTime m_timestamp{}; //!< Timestamp of of this data type
    SourceType m_sourceType{SourceType::Unknown}; //!< Indicates source of this data type
    DestinationType m_destinationType{
        DestinationType::Unknown}; //!< Indicates the type of this destination e.g. a station
    PurposeType m_purposeType{PurposeType::Unknown}; //!< Indicates the purpose of this destination e.g. maintenance
    uint32_t m_id{0}; //!< Identifier of destination, e.g. mission id

private:
    ReservedArray m_reserved{{}}; //!< Container for reserved placeholders, s. \a nbOfReserved.
}; // Destination3521

//==============================================================================

//==============================================================================
//!\brief Equal operator. Compares each member, uses 6th decimal place as floating point comparison precission.
//!\return True, if input objects are considered equal.
//------------------------------------------------------------------------------
bool operator==(const Destination3521& lhs, const Destination3521& rhs);

//==============================================================================
//!\brief Unequal operator. Negates result of equal operator.
//!\return True, if input objects are not considered equal.
//------------------------------------------------------------------------------
bool operator!=(const Destination3521& lhs, const Destination3521& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
