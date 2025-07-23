//==============================================================================
//! \file
//!
//! \brief MetaInformationList data holder for processing description.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date May 30, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationBaseIn7110.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Meta information data holder for processing description.
//------------------------------------------------------------------------------
class MetaInformationProcessingDescriptionIn7110 : public MetaInformationBaseIn7110
{
public:
    //========================================
    //! \brief Type of id data.
    //----------------------------------------
    using IdType = std::string;

public:
    //========================================
    //! \brief Default constructor.
    //!
    //! Initializes this instance as meta information of type \c ProcessingDescription.
    //----------------------------------------
    MetaInformationProcessingDescriptionIn7110();

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~MetaInformationProcessingDescriptionIn7110() override;

public:
    //========================================
    //! \brief Get the processing id.
    //! \note Will be cutted if the string is longer as \a std::numeric_limits<uint16_t>::max().
    //! \return The processing id as string.
    //----------------------------------------
    const IdType& getId() const;

    //========================================
    //! \brief Set the processing id.
    //! \note Will be cutted if the string is longer as \a std::numeric_limits<uint16_t>::max().
    //! \param[in] id  The processing id as string.
    //----------------------------------------
    void setId(const IdType& id);

public:
    //========================================
    //! \brief Tests this meta information for equality.
    //!
    //! \param[in] otherBase  The other meta information to compare with.
    //! \return \c True, if the two meta information are equal, \c false otherwise.
    //----------------------------------------
    bool isEqual(const MetaInformationBaseIn7110& otherBase) const override;

    //========================================
    //! \brief Get the size of the serialized payload.
    //!
    //! \return The size of the serialized payload.
    //----------------------------------------
    uint32_t getSerializedPayloadSize() const override;

    //========================================
    //! \brief Deserialize the data from a stream.
    //!
    //! \param[in,out] is           The stream to read the data from.
    //! \param[in]     payloadSize  The size of the payload in the stream.
    //! \return \c True, if the deserialization was successful, \c false otherwise.
    //----------------------------------------
    bool deserializePayload(std::istream& is, const uint32_t payloadSize) override;

    //========================================
    //! \brief Serialize the data into a stream.
    //!
    //! \param[in,out] os           The stream to write the data to.
    //! \return \c True, if the serialization was successful, \c false otherwise.
    //----------------------------------------
    bool serializePayload(std::ostream& os) const override;

private:
    IdType m_id;
}; // MetaInformationProcessingDescriptionIn7110

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
