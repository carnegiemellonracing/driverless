//==============================================================================
//!\file
//!
//! \brief Rtp data structures for jpeg.
//!
//! See \link https://tools.ietf.org/html/rfc2435 and \link https://datatracker.ietf.org/doc/html/rfc3550.
//!
//! \date Apr 28th, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include "microvision/common/sdk/misc/defines/defines.hpp"

//========================================
//! \brief Header for one RTP packet.
//!
//! \link https://datatracker.ietf.org/doc/html/rfc3550#section-5.1
//----------------------------------------
struct RtpHeader
{
    uint8_t version; //!< This field identifies the version of RTP. See rfc.
    uint8_t
        padding; //!< If set, the packet contains one or more additional padding octets at the end which are not part of the payload.
    uint8_t extension; //!< If set, the fixed header MUST be followed by exactly one header extension. See rfc.
    uint8_t
        csrcCount; //!< The contributing source count contains the number of contributing source identifiers. See rfc.
    uint8_t marker; //!< Defined by profile. See rfc.
    uint8_t payloadType; //!< Identifies the format of the RTP payload. See rfc.
    uint16_t seqNumber; //!< Increments by one for each RTP data packet sent.
    uint32_t timestamp; //!< Sampling instant of the first octet. See rfc.

    static constexpr uint8_t packetTypeJpeg         = 26; //!< Only this image type packet is supported.
    static constexpr uint8_t packetTypeSenderReport = 200; //!< This packet type is required for time sync.

    static constexpr const std::size_t defaultHeaderSize{1024}; //!< Size of memory for header deserialization.
    static constexpr const std::size_t maxHeaderSize{
        2048}; //!< Maximum size of memory for header deserialization if default was not large enough.
};

//========================================
//! \brief Header for a sender report packet.
//! This contains the timestamp needed to sync the time.
//! \link https://datatracker.ietf.org/doc/html/rfc3550#section-6.4.1
//----------------------------------------
struct RtpSenderReportHeader
{
    unsigned char version; //!< This field identifies the version of RTP. See rfc.
    unsigned char
        padding; //!< If set, the packet contains one or more additional padding octets at the end which are not part of the payload.
    unsigned char receptionType; //!< Number of reception report blocks. See rfc.
    unsigned char packetType; //!< Always set to 200.
    uint16_t length; //!< Length of this RTCP packet in 32-bit words minus one. See rfc.
    uint32_t senderSSRC; //!< Synchronization source identifier for the originator. See rfc.
    uint32_t ntpTimestampMSW; //!< Wall clock time most significant word when this report was sent. See rfc.
    uint32_t ntpTimestampLSW; //!< Wall clock time least significant word when this report was sent. See rfc.
    uint32_t rtpTimestamp; //!< Random rtp timestamp corresponding to wall clock time. See rfc.
    uint32_t senderPacketCount; //!< Total number of RTP data packets transmitted. See rfc.
    uint32_t senderOctetCount; //!< Total number of payload octets. See rfc.

    static constexpr std::size_t serializedHeaderSize
        = 28; //!< serialized header size is smaller than structure see rfc
};

//========================================
//! \brief Header for one chunk of a jpeg image in RTP packet.
//!
//! \link https://datatracker.ietf.org/doc/html/rfc2435
//----------------------------------------
struct RtpJpegHeader
{
    int32_t offset; //!< Fragment offset of the current packet in the JPEG frame data. See rfc.
    uint8_t type; //!< Jpeg type. See rfc.
    int32_t q; //!< Defines the quantization tables for this frame. See rfc.
    int32_t width; //!< Width of the image in 8-pixel multiples.
    int32_t height; //!< Height of the image in 8-pixel multiples.
    int32_t
        restartInterval; //!< Additional information required to properly decode a data stream containing restart markers. See rfc.
};

//========================================
//! \brief Camera info in RTP packet.
//----------------------------------------
struct CameraInfo
{
    uint16_t width  = 0; //!< Camera image width information.
    uint16_t height = 0; //!< Camera image height information.
};
