//==============================================================================
//! \file
//!
//! \brief Helper class to access/build RTSP message data.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date May 5, 2022
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/ip/IpAddress.hpp>
#include <microvision/common/sdk/io/http/HttpRequestNetworkDataPackage.hpp>
#include <microvision/common/sdk/io/http/HttpResponseNetworkDataPackage.hpp>

#include <map>
#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Helper class to access/build RTSP message data.
//------------------------------------------------------------------------------
class RtspMessageHelper final
{
public:
    //========================================
    //! \brief Type to store transport header value in key/value pairs.
    //----------------------------------------
    using TransportInfoType = std::map<std::string, std::string>;

    //========================================
    //! \brief Type to store DESCRIBE body in key/value pairs.
    //----------------------------------------
    using DescribeInfoType = std::map<std::string, std::list<std::string>>;

public:
    //========================================
    //! \brief HTTP version of RTSP 1.0.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string version1p0;

    //========================================
    //! \brief HTTP url scheme of RTSP 1.0.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string version1p0UrlSchema;

    //========================================
    //! \brief HTTP request method 'OPTIONS' to get supported RTSP request methods.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string optionsRequestMethod;

    //========================================
    //! \brief HTTP request method 'DESCRIBE' to get RTSP stream meta information.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string describeRequestMethod;

    //========================================
    //! \brief HTTP request method 'SETUP' to create a new stream instance.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string setupRequestMethod;

    //========================================
    //! \brief HTTP request method 'TEARDOWN' to delete a stream instance.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string teardownRequestMethod;

    //========================================
    //! \brief HTTP request method 'PLAY' to start stream data via UDP.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string playRequestMethod;

    //========================================
    //! \brief HTTP request method 'PAUSE' to stop stream data via UDP.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string pauseRequestMethod;

    //========================================
    //! \brief HTTP request method 'PAUSE' to get device configuration.
    //! \note Will also use to keep the connection alive.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string getParameterRequestMethod;

    //========================================
    //! \brief HTTP header key for command sequence number.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string commandSequenceHeaderKey;

    //========================================
    //! \brief HTTP header key for accept content type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string acceptHeaderKey;

    //========================================
    //! \brief HTTP header key for content base URL.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string contentBaseHeaderKey;

    //========================================
    //! \brief HTTP header key for content location URL of resource.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string contentLocationHeaderKey;

    //========================================
    //! \brief HTTP header key for content type like 'application/sdp'.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string contentTypeHeaderKey;

    //========================================
    //! \brief HTTP header key for content length.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string contentLengthHeaderKey;

    //========================================
    //! \brief HTTP header key for transport configuration like ip and port etc.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string transportHeaderKey;

    //========================================
    //! \brief HTTP header key for stream session id.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string sessionHeaderKey;

    //========================================
    //! \brief HTTP header key for stream range.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string rangeHeaderKey;

    //========================================
    //! \brief HTTP header key for RTP stream information.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string rtpInfoHeaderKey;

    //========================================
    //! \brief Default accept header value.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string defaultAcceptHeaderValue;

    //========================================
    //! \brief Default range header value to start from begin.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string defaultRangeHeaderValue;

    //========================================
    //! \brief Control key of DESCRIBE body to compile content URL for stream request.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string describeBodyControlKey;

    //========================================
    //! \brief Control value of DESCRIBE body which represents the placeholder for using the content base path.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string describeBodyControlValue;

    //========================================
    //! \brief SETUP transport protocol for RTP image stream.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string setupTransportProtocol;

    //========================================
    //! \brief SETUP transport type for UDP unicast.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string setupTransportType;

    //========================================
    //! \brief Transport source key.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string transportSource;

    //========================================
    //! \brief Transport destination key.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string transportDestination;

    //========================================
    //! \brief Transport client port key.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string transportClientPort;

    //========================================
    //! \brief Transport server port key.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string transportServerPort;

public:
    //========================================
    //! \brief Parse DESCRIBE body into key/value pairs.
    //! \param[in]  body            Body of HTTP message.
    //! \param[out] describeInfo    Parsed key/value pairs.
    //! \return Either \c true if successful parsed otherwise \c false.
    //----------------------------------------
    static bool parseDescribeBody(const SharedBuffer& body, DescribeInfoType& describeInfo);

    //========================================
    //! \brief Parse transport header into key/value pairs.
    //! \param[in]  package         HTTP message data package.
    //! \param[out] transportInfo   Parsed key/value pairs.
    //! \return Either \c true if successful parsed otherwise \c false.
    //----------------------------------------
    static bool parseTransportParameter(const HttpNetworkDataPackage& package, TransportInfoType& transportInfo);

    //========================================
    //! \brief Set transport header after stringifying key/value pairs.
    //! \param[in, out]  package   HTTP message data package.
    //! \param[in] transportInfo   New transport info as key/value pairs.
    //----------------------------------------
    static void setTransportParameter(HttpNetworkDataPackage& package, const TransportInfoType& transportInfo);

    //========================================
    //! \brief Parse rtp/rtcp port set which is separated by '-'.
    //! \param[in]  ports       Rtp and rtcp port separated by '-'.
    //! \param[out] rtpPort     Parsed rtp port.
    //! \param[out] rtcpPort    Parsed rtcp port.
    //! \return Either \c true if successful parsed otherwise \c false.
    //----------------------------------------
    static bool parsePortSet(const std::string& ports, uint16_t& rtpPort, uint16_t& rtcpPort);

    //========================================
    //! \brief Stringify rtp/rtcp port set which is separated by '-'.
    //! \param[in] rtpPort     Rtp port.
    //! \param[in] rtcpPort    Rtcp port.
    //! \return Concatenated rtp and rtcp port separated by '-'.
    //----------------------------------------
    static std::string stringifyPortSet(const uint16_t& rtpPort, const uint16_t& rtcpPort);

}; // class RtspMessageHelper

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
