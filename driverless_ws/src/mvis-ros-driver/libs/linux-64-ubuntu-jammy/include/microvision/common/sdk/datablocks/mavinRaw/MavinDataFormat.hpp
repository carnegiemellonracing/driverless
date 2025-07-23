//==============================================================================
//! \file
//!
//! \brief Data type to store mavin raw frame format extracted from header data.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 27th, 2023
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/Optional.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Enum for the MAVIN sensor tcp stream format.
//------------------------------------------------------------------------------
enum class MavinDataFormat : uint8_t
{
    None   = 0,
    MVO    = 1,
    PCRaw  = 2,
    CueTDC = 3
};

//==============================================================================
//! \brief Make enum value of \a MavinDataFormat from string value.
//! \param[in] value  String value of \a MavinDataFormat enum.
//! \return Enum value of \a MavinDataFormat.
//! \note If the conversion failed the enum value \c None will returned.
//------------------------------------------------------------------------------
MavinDataFormat makeMavinDataFormat(const std::string& value);

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief MAVIN sensor major, minor and patch version parts.
//------------------------------------------------------------------------------
class MavinDataFormatVersion final
{
public:
    using VersionArray = std::array<uint16_t, 3>; //!< Array of version parts.

    static constexpr std::size_t majorVersionIdx{0}; //!< Index of major version in version array
    static constexpr std::size_t minorVersionIdx{1}; //!< Index of minor version in version array
    static constexpr std::size_t patchVersionIdx{2}; //!< Index of patch version in version array

public:
    //========================================
    //! \brief Default constructor. Sets all version parts to 0.
    //----------------------------------------
    MavinDataFormatVersion();

    //========================================
    //! \brief Constructs a MAVIN sensor version.
    //! \param[in] majorVersion  The major version part.
    //! \param[in] minorVersion  (Optional) The minor version part. (Default: 0)
    //! \param[in] patchVersion  (Optional) The patch version part. (Default: 0)
    //----------------------------------------
    MavinDataFormatVersion(const uint16_t majorVersion,
                           const uint16_t minorVersion = 0,
                           const uint16_t patchVersion = 0);

public:
    //========================================
    //! \brief Get the version parts (major, minor, patch).
    //----------------------------------------
    const VersionArray& getVersion() const;

    //========================================
    //! \brief Get the major version.
    //----------------------------------------
    uint16_t getMajorVersion() const;

    //========================================
    //! \brief Get the minor version.
    //----------------------------------------
    uint16_t getMinorVersion() const;

    //========================================
    //! \brief Get the patch version.
    //----------------------------------------
    uint16_t getPatchVersion() const;

    //========================================
    //! \brief Checks whether version is invalid (0.0.0).
    //! \return Either \c true if invalid or otherwise \c false.
    //----------------------------------------
    bool isInvalid() const;

private:
    VersionArray m_version;
}; // MavinDataFormatVersion

//==============================================================================
//! \brief Checks whether \a lhs is smaller than \a rhs.
//!
//! Lexicographical order is applied.
//!
//! \param[in] lhs  The left version.
//! \param[in] rhs  The right version.
//! \return \c true, if lhs is smaller than rhs. \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator<(const MavinDataFormatVersion& lhs, const MavinDataFormatVersion rhs)
{
    return (lhs.getVersion() < rhs.getVersion()); // using lexipographical order of std::array
}

//==============================================================================
//! \brief Checks whether \a lhs is equal to \a rhs.
//!
//! \param[in] lhs  The left version.
//! \param[in] rhs  The right version.
//! \return \c true, if lhs is equal to rhs. \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator==(const MavinDataFormatVersion& lhs, const MavinDataFormatVersion rhs)
{
    return (lhs.getVersion() == rhs.getVersion());
}

//==============================================================================

//==============================================================================
//! \brief Make \a MavinDataFormatVersion from version string \a versionText.
//!
//! The MavinDataFormatVersion is divided into three parts representing major,
//! minor and patch version.
//!
//! - While the major part is mandatory, minor and patch are optional and will be
//!   treated as 0 if not present.
//! - The parts in \a versionText have to be separated by a .
//! - Each part of the version has to represent a value between 0 and 9999.
//! - A version 0.0.0 means invalid.
//!
//! \example "1.2.3"  or "1.2" or "1" are valid inputs.
//!
//! \param[in] versionText  The string that contains the version.
//!
//! \return \a MavinDataFormatVersion
//------------------------------------------------------------------------------
MavinDataFormatVersion makeMavinDataFormatVersion(const std::string& versionText);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
namespace std {
//==============================================================================

//==============================================================================
//! \brief Make a string of \a MavinDataFormat enum value.
//! \param[in] value  Enum value of \a MavinDataFormat.
//! \return String value of \a MavinDataFormat.
//------------------------------------------------------------------------------
std::string to_string(const microvision::common::sdk::MavinDataFormat value);

//==============================================================================
//! \brief Make a string of \a MavinDataFormatVersion class.
//! \param[in] value  Instance of \a MavinDataFormatVersion.
//! \return String value of \a MavinDataFormatVersion.
//------------------------------------------------------------------------------
std::string to_string(const microvision::common::sdk::MavinDataFormatVersion version);

//==============================================================================
} // namespace std
//==============================================================================
