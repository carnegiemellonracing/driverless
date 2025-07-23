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

#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationBaseIn7110.hpp>

#include <string>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class holds a version number and additional information about
//! an application used during this recording.
//------------------------------------------------------------------------------
class MetaInformationVersionNumberIn7110 final : public MetaInformationBaseIn7110
{
public:
    //========================================
    //! \brief Type of MicroVision application.
    //----------------------------------------
    enum class SoftwareType : uint16_t
    {
        Undefined           = 0, //!< Undefined application is used.
        Custom              = 1, //!< A custom application is used.
        AppBase             = 2, //!< The AppBase is used.
        EvS                 = 3, //!< The EVS is used.
        Ilv                 = 4, //!< The ILV is used.
        SyncBox             = 5, //!< The SyncBox is used.
        VpcapToIdcConverter = 6 //!< The VPCAP2IDC Converter is used.
    };

    using VersionNumberType = uint32_t;
    using VersionPartType   = uint16_t;

public:
    //========================================
    //! \brief Constructor from version number.
    //!
    //! Initializes this instance as meta information of type version number.
    //----------------------------------------
    MetaInformationVersionNumberIn7110()
      : MetaInformationBaseIn7110(MetaInformationBaseIn7110::MetaInformationType::VersionNumber)
    {}

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    virtual ~MetaInformationVersionNumberIn7110() = default;

public: // getter
    //========================================
    //! \brief Get the version number.
    //!
    //! \return The version number.
    //----------------------------------------
    VersionNumberType getVersionNumber() const { return m_versionNumber; }

    //========================================
    //! \brief Get the major field of the version number.
    //!
    //! \return The major field of the version number.
    //----------------------------------------
    VersionPartType getMajorVersion() const;

    //========================================
    //! \brief Get the minor field of the version number.
    //!
    //! \return The minor field of the version number.
    //----------------------------------------
    VersionPartType getMinorVersion() const;

    //========================================
    //! \brief Get the patch field of the version number.
    //!
    //! \return The patch field of the version number.
    //----------------------------------------
    VersionPartType getPatchVersion() const;

    //========================================
    //! \brief Get the extra field of the version number.
    //!
    //! \return The extra field of the version number.
    //----------------------------------------
    const std::string& getExtraString() const { return m_extraString; }

    //========================================
    //! \brief Get the software type.
    //!
    //! \return The software type.
    //----------------------------------------
    SoftwareType getSoftwareType() const { return m_softwareType; }

public: // setter
    //========================================
    //! \brief Set the version number.
    //!
    //! \param[in] newVersion  The new version number.
    //----------------------------------------
    void setVersionNumber(const uint32_t newVersion) { m_versionNumber = newVersion; }

    //========================================
    //! \brief Set the version number.
    //!
    //! \param[in] major  The major field of the new version number.
    //! \param[in] minor  The major field of the new version number.
    //! \param[in] patch  The patch field of the new version number.
    //! \param[in] extra  The extra field of the new version number.
    //----------------------------------------
    void
    setVersionNumber(const uint16_t major, const uint16_t minor, const uint16_t patch, const std::string extra = "");

    //========================================
    //! \brief Set the extra field of the version number.
    //!
    //! \param[in] extra  The extra field of the new version number.
    //----------------------------------------
    void setExtraString(const std::string& extra) { m_extraString = extra; }

    //========================================
    //! \brief Set the software type.
    //!
    //! \param[in] type  The new software type.
    //----------------------------------------
    void setSoftwareType(const SoftwareType type) { m_softwareType = type; }

public: // MetaInformationBaseIn7110 interface
    //========================================
    //! \brief Tests this meta information for equality.
    //!
    //! \param[in] otherBase  The other meta information to compare with.
    //! \return \c True, if the two meta information are equal, \c false otherwise.
    //----------------------------------------
    virtual bool isEqual(const MetaInformationBaseIn7110& other) const override;

    //========================================
    //! \brief Get the size of the serialized payload.
    //!
    //! \return The size of the serialized payload.
    //----------------------------------------
    virtual uint32_t getSerializedPayloadSize() const override;

    //========================================
    //! \brief Deserialize the data from a stream.
    //!
    //! \param[in,out] is           The stream to read the data from.
    //! \param[in]     payloadSize  The size of the payload in the stream.
    //! \return \c True, if the deserialization was successful, \c false otherwise.
    //----------------------------------------
    virtual bool deserializePayload(std::istream& is, const uint32_t payloadSize) override;

    //========================================
    //! \brief Serialize the data into a stream.
    //!
    //! \param[in,out] os           The stream to write the data to.
    //! \return \c True, if the serialization was successful, \c false otherwise.
    //----------------------------------------
    virtual bool serializePayload(std::ostream& os) const override;

private:
    SoftwareType m_softwareType{SoftwareType::Undefined}; //!< The software type.

    //! \brief The version number.
    //!
    //! \details
    //! Major: VersionNumber/1000000
    //! Minor: (VersionNumber%1000000)/1000
    //! Patch: VersionNumber%1000
    VersionNumberType m_versionNumber{0};

    std::string m_extraString{}; //!<  Custom version string.
}; // MetaInformationVersionNumberIn7110

//==============================================================================

std::ostream& operator<<(std::ostream& os, const MetaInformationVersionNumberIn7110::SoftwareType st);

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
