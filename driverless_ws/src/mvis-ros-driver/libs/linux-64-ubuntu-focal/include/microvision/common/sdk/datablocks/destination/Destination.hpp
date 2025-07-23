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
//! \date Jan 28, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>
#include <microvision/common/sdk/datablocks/destination/special/Destination3520.hpp>
#include <microvision/common/sdk/datablocks/destination/special/Destination3521.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================
//! \brief General data container for destination.
//!
//! target position as GPS coordinate
//! this data type was generated with the interface to BestMile for fleet management
//! in mind [https://developer.bestmile.com/v1/docs/performing-a-mission#section-mission-fields-overview],
//! it's similar to Destination field
//------------------------------------------------------------------------------
class Destination final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const Destination& lhs, const Destination& rhs);

public: // type declaration
    using ReservedArray   = Destination3521::ReservedArray;
    using SourceType      = Destination3521::SourceType;
    using DestinationType = Destination3521::DestinationType;
    using PurposeType     = Destination3521::PurposeType;

public:
    //==============================================================================
    //! \brief Empty constructor, initializes \a m_delegate with empty constructor call.
    //------------------------------------------------------------------------------
    Destination();

    //==============================================================================
    //! \brief Default destructor.
    //------------------------------------------------------------------------------
    ~Destination() override = default;

public:
    constexpr static const char* const containerType{
        "sdk.generalcontainer.destination"}; //!< Statically defined type of container.

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
    //! \brief Trivial setter. Sets input argument to respective value in \a m_delegate container.
    //! \param[in] targetPosition  Goal position of target in wgs84 format.
    //------------------------------------------------------------------------------
    void setTargetPosition(const PositionWgs84& targetPosition) { m_delegate.setTargetPosition(targetPosition); }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective value in \a m_delegate container.
    //! \param[in] targetHeading  Global orientation of target in radian.
    //------------------------------------------------------------------------------
    void setTargetHeading(const double targetHeading) { m_delegate.setTargetHeading(targetHeading); }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective value in \a m_delegate container.
    //! \param[in] targetId  Id of target.
    //------------------------------------------------------------------------------
    void setTargetId(const int64_t targetId) { m_delegate.setTargetId(targetId); }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective value in \a m_delegate container.
    //! \param[in] timestamp  Time at which this message was generated / sent.
    //------------------------------------------------------------------------------
    void setTimestamp(const NtpTime& timestamp) { m_delegate.setTimestamp(timestamp); }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective value in \a m_delegate container.
    //! \param[in] sourceType  Defines source or context this message was generated in. See type declarations at top of
    //!                        class for options.
    //------------------------------------------------------------------------------
    void setSourceType(const SourceType sourceType) { m_delegate.setSourceType(sourceType); }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective value in \a m_delegate container.
    //! \param[in] destinationType  Defines the type of destination. See type declarations at top of class for options.
    //------------------------------------------------------------------------------
    void setDestinationType(const DestinationType destinationType) { m_delegate.setDestinationType(destinationType); }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective value in \a m_delegate container.
    //! \param[in] purposeType  Defines the purpose in context of missions. See type declarations at
    //!                         top of class for options.
    //------------------------------------------------------------------------------
    void setPurposeType(const PurposeType purposeType) { m_delegate.setPurposeType(purposeType); }

    //==============================================================================
    //! \brief Trivial setter. Sets input argument to respective value in \a m_delegate container.
    //! \param[in] id  Id of this message.
    //------------------------------------------------------------------------------
    void setId(const uint32_t id) { m_delegate.setId(id); }

private:
    //==============================================================================
    //! \brief Sets attributes of \a m_delegate to values of input argument.
    //! \param[in] destination  Contains attributes to be copied to m_delegate.
    //------------------------------------------------------------------------------
    void setAttributesFrom(const Destination3520& destination);

public:
    //==============================================================================
    //! \brief Trivial getter. Returns respective member of \a m_delegate container.
    //! \return Position of the target in wgs84 format.
    //------------------------------------------------------------------------------
    const PositionWgs84& getTargetPosition() const { return m_delegate.getTargetPosition(); }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member of \a m_delegate container.
    //! \return Global orientation of the target in radian.
    //------------------------------------------------------------------------------
    double getTargetHeading() const { return m_delegate.getTargetHeading(); }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member of \a m_delegate container.
    //! \return Id of the target.
    //------------------------------------------------------------------------------
    int64_t getTargetId() const { return m_delegate.getTargetId(); }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member of \a m_delegate container.
    //! \return Time at which the message was generated / sent.
    //------------------------------------------------------------------------------
    const NtpTime& getTimestamp() const { return m_delegate.getTimestamp(); }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member of \a m_delegate container.
    //! \return Source or context this message was generated in. See type declarations at top of class for options.
    //------------------------------------------------------------------------------
    SourceType getSourceType() const { return m_delegate.getSourceType(); }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member of \a m_delegate container.
    //! \return Type of destination. See type declarations at top of class for options.
    //------------------------------------------------------------------------------
    DestinationType getDestinationType() const { return m_delegate.getDestinationType(); }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member of \a m_delegate container.
    //! \return Purpose in context of missions. See type declarations at top of class for options.
    //------------------------------------------------------------------------------
    PurposeType getPurposeType() const { return m_delegate.getPurposeType(); }

    //==============================================================================
    //! \brief Trivial getter. Returns respective member of \a m_delegate container.
    //! \return Id of this message.
    //------------------------------------------------------------------------------
    uint32_t getId() const { return m_delegate.getId(); }

    //==============================================================================
    //!\brief Returns value from \a m_reserved array at specified position \a idx.
    //!\param[in] idx  Index specifying position in \a m_reserved array.
    //!\return Value of \a m_reserved array at position \a idx.
    //!        Returns last element of array, if \a idx >= \a nbOfReserved.
    //------------------------------------------------------------------------------
    uint32_t getReserved(const uint8_t idx) const { return m_delegate.getReserved(idx); }

private:
    //==============================================================================
    //! \brief Copies attributes of \a m_delgate to output argument.
    //! \param[out] destination  Filled with attributes of \a m_delegate.
    //------------------------------------------------------------------------------
    void copyAttributesTo(Destination3520& destination) const;

protected:
    Destination3521 m_delegate; //!< Delegate holding actual data of this data container.
}; //DestinationContainer
//==============================================================================

//==============================================================================
//!\brief Equal operator. Uses equal operator of \a m_delegate.
//!\return True, if input objects are considered equal.
//------------------------------------------------------------------------------
bool operator==(const Destination& lhs, const Destination& rhs);

//==============================================================================
//!\brief Unequal operator. Negates result of equal operator.
//!\return True, if input objects are not considered equal.
//------------------------------------------------------------------------------
bool operator!=(const Destination& lhs, const Destination& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
