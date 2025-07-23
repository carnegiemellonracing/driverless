//==============================================================================
//!\file
//!
//! \brief Abstract data package base for HTTP protocol.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 06, 2022
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/NetworkDataPackage.hpp>

#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Abstract data package base for HTTP protocol.
//!
//! For more information see: \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview
//!
//! \extends NetworkDataPackage
//------------------------------------------------------------------------------
class HttpNetworkDataPackage : public NetworkDataPackage
{
public:
    //========================================
    //! \brief Default HTTP protocol version.
    //----------------------------------------
    static constexpr const char* defaultVersion{"HTTP/1.1"};

    //========================================
    //! \brief Key/Value map to store HTTP headers.
    //----------------------------------------
    using HeaderMapType = std::map<std::string, std::string>;

protected:
    //========================================
    //! \brief Default constructor.
    //! \param[in] index            Position in the data package stream, depends on source.
    //! \param[in] sourceUri        Source Uri of data (stream).
    //! \param[in] destinationUri   Destination Uri of data (stream).
    //! \param[in] method           Request method.
    //! \param[in] path             Request path.
    //! \param[in] headers          Request header.
    //! \param[in] payload          Request body.
    //----------------------------------------
    HttpNetworkDataPackage(const int64_t index          = 0,
                           const Uri& sourceUri         = Uri{},
                           const Uri& destinationUri    = Uri{},
                           const HeaderMapType& headers = {},
                           const SharedBuffer& payload  = {});

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~HttpNetworkDataPackage() override;

public:
    //========================================
    //! \brief Compare two HTTP data packages for equality.
    //! \param[in] lhs  HttpNetworkDataPackage to compare.
    //! \param[in] rhs  HttpNetworkDataPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const HttpNetworkDataPackage& lhs, const HttpNetworkDataPackage& rhs);

    //========================================
    //! \brief Compare two HTTP data packages for inequality.
    //! \param[in] lhs  HttpNetworkDataPackage to compare.
    //! \param[in] rhs  HttpNetworkDataPackage to compare.
    //! \note Offset wont compare because of section compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const HttpNetworkDataPackage& lhs, const HttpNetworkDataPackage& rhs);

public: //getter
    //========================================
    //! \brief Get protocol version.
    //! \return Protocol version as string.
    //----------------------------------------
    const std::string& getVersion() const;

    //========================================
    //! \brief Get map of header informations.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers
    //!
    //! \return Header as map.
    //----------------------------------------
    HeaderMapType& getHeaders();

    //========================================
    //! \brief Get map of header informations.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers
    //!
    //! \return Header as map.
    //----------------------------------------
    const HeaderMapType& getHeaders() const;

public: //setter
    //========================================
    //! \brief Set protocol version.
    //! \param[in] version  Protocol version as string.
    //----------------------------------------
    void setVersion(const std::string& version);

    //========================================
    //! \brief Get map of header information's.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers
    //!
    //! \param[in] headers  Header as map.
    //----------------------------------------
    void setHeaders(const HeaderMapType& headers);

private:
    //========================================
    //! \brief Protocol version.
    //----------------------------------------
    std::string m_version;

    //========================================
    //! \brief Map of header informations.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers
    //----------------------------------------
    HeaderMapType m_headers;

}; // class HttpNetworkDataPackage

//==============================================================================
//! \brief Nullable HttpNetworkDataPackage pointer.
//------------------------------------------------------------------------------
using HttpNetworkDataPackagePtr = std::shared_ptr<HttpNetworkDataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
