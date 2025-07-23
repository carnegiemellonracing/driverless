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
//! \date Mar 04, 2020
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/Utils.hpp>
#include <microvision/common/sdk/misc/Uri.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Static class to provide helper functions about uris.
//------------------------------------------------------------------------------
class IdcUriHelper final
{
public:
    //========================================
    //! \brief Name of the IDC format for the Uri.
    //----------------------------------------
    static constexpr const char* IDC_FORMAT_NAME{"IDC"};

    //========================================
    //! \brief Name of the IDC schema for the Uri.
    //----------------------------------------
    static constexpr const char* IDC_SCHEMA_NAME{"IDC"};

    //========================================
    //! \brief File name extension for IDC files.
    //----------------------------------------
    static constexpr const char* IDC_FILENAME_EXTENSION{".idc"};

public:
    //========================================
    //! \brief Either \c true if the uri represent a file or
    //!         address less uri of schema IDC, otherwise \c false.
    //! \param[in] path  Uri to check.
    //! \return Either \c true if the schema of uri is IDC, otherwise \c false.
    //----------------------------------------
    static bool hasIdcSchema(const Uri& path)
    {
        const bool formatCheck = compare(path.getFormat(), std::string{IDC_FORMAT_NAME}, true)
                                 || compare(path.getSchema(), std::string{IDC_SCHEMA_NAME}, true);

        return ((path.getProtocol() == UriProtocol::NoAddr) && formatCheck)
               || ((path.getProtocol() == UriProtocol::File)
                   && (formatCheck || endsWith(path.getPath(), std::string{IDC_FILENAME_EXTENSION}, true)));
    }

    //========================================
    //! \brief Either \c true if the uri represent a file or
    //!         address less uri of format IDC, otherwise \c false.
    //! \param[in] path  Uri to check.
    //! \return Either \c true if the format of uri is IDC, otherwise \c false.
    //----------------------------------------
    static bool hasIdcFormat(const Uri& path)
    {
        const bool formatCheck = compare(path.getFormat(), std::string{IDC_FORMAT_NAME}, true);

        return ((path.getProtocol() == UriProtocol::NoAddr) && formatCheck)
               || ((path.getProtocol() == UriProtocol::File)
                   && (formatCheck || endsWith(path.getPath(), std::string{IDC_FILENAME_EXTENSION}, true)));
    }

    //========================================
    //! \brief Create a file Uri as idc schema from file path.
    //! \param[in] filePath  Valid file path of system.
    //! \return Created Uri of protocol file and idc schema.
    //----------------------------------------
    inline static Uri createIdcSchemaFileUri(const std::string& filePath)
    {
        Uri result{filePath};

        result.setSchema(IDC_SCHEMA_NAME);

        return result;
    }

    //========================================
    //! \brief Create a file Uri as idc format from file path.
    //! \param[in] filePath  Valid file path of system.
    //! \return Created Uri of protocol file and idc format.
    //----------------------------------------
    inline static Uri createIdcFormatFileUri(const std::string& filePath)
    {
        Uri result{filePath};

        result.setFormat(IDC_FORMAT_NAME);
        result.setSchema(IDC_SCHEMA_NAME);

        return result;
    }
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
