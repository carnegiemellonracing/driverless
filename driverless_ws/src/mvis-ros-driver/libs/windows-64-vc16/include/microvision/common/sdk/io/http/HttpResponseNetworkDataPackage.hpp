//==============================================================================
//!\file
//!
//! \brief Defines data package for HTTP response protocol.
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
//! \brief Data package for HTTP response protocol.
//! \extends NetworkDataPackage
//------------------------------------------------------------------------------
class HttpResponseNetworkDataPackage final : public HttpNetworkDataPackage
{
public:
    //========================================
    //! \brief The status code which identified the request as successful.
    //----------------------------------------
    static constexpr uint16_t statusCodeOk{200U};

public:
    //========================================
    //! \brief Response constructor.
    //! \param[in] index            Position in the data package stream, depends on source.
    //! \param[in] sourceUri        Source Uri of data (stream).
    //! \param[in] destinationUri   Destination Uri of data (stream).
    //! \param[in] statusCode       Response status code.
    //! \param[in] statusMessage    Response status message.
    //! \param[in] headers          Response header.
    //! \param[in] payload          Response body.
    //----------------------------------------
    HttpResponseNetworkDataPackage(const int64_t index              = 0,
                                   const Uri& sourceUri             = Uri{},
                                   const Uri& destinationUri        = Uri{},
                                   const uint16_t& statusCode       = 0,
                                   const std::string& statusMessage = "",
                                   const HeaderMapType& headers     = {},
                                   const SharedBuffer& payload      = {});

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~HttpResponseNetworkDataPackage() override;

public:
    //========================================
    //! \brief Compare two HTTP data packages for equality.
    //! \param[in] lhs  HttpResponseNetworkDataPackage to compare.
    //! \param[in] rhs  HttpResponseNetworkDataPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const HttpResponseNetworkDataPackage& lhs, const HttpResponseNetworkDataPackage& rhs);

    //========================================
    //! \brief Compare two HTTP data packages for inequality.
    //! \param[in] lhs  HttpResponseNetworkDataPackage to compare.
    //! \param[in] rhs  HttpResponseNetworkDataPackage to compare.
    //! \note Offset wont compare because of section compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const HttpResponseNetworkDataPackage& lhs, const HttpResponseNetworkDataPackage& rhs);

public: //getter
    //========================================
    //! \brief Get respone status code.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    //!
    //! \return Status code as number.
    //----------------------------------------
    const uint16_t& getStatusCode() const;

    //========================================
    //! \brief Get response status message.
    //! \return Status message as string.
    //----------------------------------------
    const std::string& getStatusMessage() const;

public: //setter
    //========================================
    //! \brief Set response status code.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    //!
    //! \param[in] statusCode  Status code as number.
    //----------------------------------------
    void setStatusCode(const uint16_t& statusCode);

    //========================================
    //! \brief Set response status message.
    //! \param[in] statusMessage  Status message as string.
    //----------------------------------------
    void setStatusMessage(const std::string& statusMessage);

private:
    //========================================
    //! \brief Response status code.
    //!
    //! For more information see \link https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    //!
    //! \note Only relevant if the package represents a response.
    //----------------------------------------
    uint16_t m_statusCode;

    //========================================
    //! \brief Response status message.
    //! \note Only relevant if the package represents a response.
    //----------------------------------------
    std::string m_statusMessage;

}; // class HttpResponseNetworkDataPackage

//==============================================================================
//! \brief Nullable HttpResponseNetworkDataPackage pointer.
//------------------------------------------------------------------------------
using HttpResponseNetworkDataPackagePtr = std::shared_ptr<HttpResponseNetworkDataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
