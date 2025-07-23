//==============================================================================
//!\file
//!
//! \brief Defines data package for HTTP request protocol.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 22, 2022
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/http/HttpNetworkDataPackage.hpp>

#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data package for HTTP request protocol.
//! \extends HttpNetworkDataPackage
//------------------------------------------------------------------------------
class HttpRequestNetworkDataPackage final : public HttpNetworkDataPackage
{
public:
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
    HttpRequestNetworkDataPackage(const int64_t index          = 0,
                                  const Uri& sourceUri         = Uri{},
                                  const Uri& destinationUri    = Uri{},
                                  const std::string& method    = "",
                                  const std::string& path      = "",
                                  const HeaderMapType& headers = {},
                                  const SharedBuffer& payload  = {});

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~HttpRequestNetworkDataPackage() override;

public:
    //========================================
    //! \brief Compare two HTTP data packages for equality.
    //! \param[in] lhs  HttpRequestNetworkDataPackage to compare.
    //! \param[in] rhs  HttpRequestNetworkDataPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const HttpRequestNetworkDataPackage& lhs, const HttpRequestNetworkDataPackage& rhs);

    //========================================
    //! \brief Compare two HTTP data packages for inequality.
    //! \param[in] lhs  HttpRequestNetworkDataPackage to compare.
    //! \param[in] rhs  HttpRequestNetworkDataPackage to compare.
    //! \note Offset wont compare because of section compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const HttpRequestNetworkDataPackage& lhs, const HttpRequestNetworkDataPackage& rhs);

public: //getter
    //========================================
    //! \brief Get request method.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
    //!
    //! \return Method as string.
    //----------------------------------------
    const std::string& getMethod() const;

    //========================================
    //! \brief Get request path/URL.
    //! \return Path/URL as string.
    //----------------------------------------
    const std::string& getPath() const;

public: //setter
    //========================================
    //! \brief Set request method.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
    //!
    //! \param[in] method  Method as string.
    //----------------------------------------
    void setMethod(const std::string& method);

    //========================================
    //! \brief Set request path/URL.
    //! \param[in] path  Path/URL as string.
    //----------------------------------------
    void setPath(const std::string& path);

private:
    //========================================
    //! \brief Request method.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
    //----------------------------------------
    std::string m_method;

    //========================================
    //! \brief Request path/URL.
    //! \note Only relevant if the package represents a request.
    //----------------------------------------
    std::string m_path;

}; // class HttpRequestNetworkDataPackage

//==============================================================================
//! \brief Nullable HttpRequestNetworkDataPackage pointer.
//------------------------------------------------------------------------------
using HttpRequestNetworkDataPackagePtr = std::shared_ptr<HttpRequestNetworkDataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
