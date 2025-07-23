//==============================================================================
//!\file
//!
//! \brief Definition of rtp packets parser for camera.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 28th, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/DataPackage.hpp>
#include <microvision/common/sdk/misc/SharedBufferStream.hpp>

#include <microvision/common/sdk/listener/DataStreamer.hpp>

#include <microvision/common/sdk/jpegsupport/Rtp.hpp>

#include <cstdint>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Functionality for parsing RTP protocol.
//!
//!         RTP: A Transport Protocol for Real-Time Applications
//!         \link https://tools.ietf.org/html/rfc3550
//!
//!         RTP Payload Format for JPEG-compressed Video
//!         \link https://tools.ietf.org/html/rfc2435
//!
//==============================================================================
class RtpParser
{
public:
    static constexpr uint8_t supportedRtpVersion = 2; //!< Only version of rtp that is currently supported.

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    explicit RtpParser();

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~RtpParser() = default;

public:
    //========================================
    //! \brief The payload of an RTP packet is stored here to be reassembled to an image.
    //----------------------------------------
    struct RtpPayload
    {
        SharedBuffer imageData; //!< Payload containing jpeg image content chunk.
        bool isNewImage; //!< Marker for start chunk of new image.
        bool isEndOfImage; //!< Marker for final chunk of an image.
        uint32_t timestamp; //!< Timestamp of image chunk from RTP packet header.
    };

public:
    // aliases
    using RtpPayloadPtr            = std::shared_ptr<RtpPayload>; //!< Pointer to RTP payload.
    using RtpSenderReportHeaderPtr = std::shared_ptr<RtpSenderReportHeader>; //!< Pointer to RTP sender response header.

public:
    //========================================
    //! \brief Get the rtp payload.
    //!
    //! Cuts rtp header from data and adjusts data length.
    //!	Warning : This function can only guess if there is a rtp header,
    //!	          because it is not stated in other packet headers(IP, UDP),
    //!	          if the rtp protocol is included.
    //!
    //! \param[in] dataPackage  DataPackage to get a RTP payload from.
    //! \return RTP data from the package; \c nullptr if none found.
    //----------------------------------------
    RtpPayloadPtr getRtpPayload(const DataPackagePtr& dataPackage);

    //========================================
    //! \brief Get a RTP sender report header for time sync if there is one in this package.
    //!
    //! \param[in] package  DataPackage to get a RTP sender report header from.
    //! \return RTP sender report header from the package; \c nullptr if none found.
    //----------------------------------------
    RtpSenderReportHeaderPtr getRtpSenderReportHeader(const DataPackagePtr& package);

    //========================================
    //! \brief Get the jpeg header.
    //!
    //! \return Returns the jpeg header bytes.
    //----------------------------------------
    const std::vector<uint8_t>& getJpegHeader() const;

    //========================================
    //! \brief Get the camera info.
    //!
    //! \return Returns the camera info.
    //----------------------------------------
    const CameraInfo& getCameraInfo() const;

    //========================================
    //! \brief Check if the RTP packet timestamp is the same as the image timestamp.
    //!
    //! \return Either \c true if the timestamps are equal or \c false if not.
    //----------------------------------------
    bool rtpTimestampEqualsImageTimestamp() const;

    //========================================
    //! \brief Get the current RTP timestamp of the running parser parsing several packets.
    //!
    //! \return The current timestamp.
    //----------------------------------------
    uint32_t getCurrentRtpTimestamp() const;

private:
    //========================================
    //! \brief Parse the RTP header from the given memory.
    //! \note No validity checks of the data.
    //!
    //! \param[in, out] stream  Buffer to read from.
    //! \param[out] rtpHeader   RTP header parsed.
    //----------------------------------------
    void parseRtpHeader(SharedBufferStream& stream, RtpHeader& rtpHeader) const;

    //========================================
    //! \brief Parse the RTP sender response from the given memory.
    //!
    //! This header is needed for the timestamp.
    //! \note No validity checks of the data.
    //!
    //! \param[in, out] stream  Buffer to read from.
    //! \param[out] rtpHeader   RTP sender response header parsed.
    //----------------------------------------
    void parseRtpSenderResponseHeader(SharedBufferStream& stream, RtpSenderReportHeader& rtpHeader);

    //========================================
    //! \brief Skip the RTP extension header in the given memory.
    //!
    //! \param[in, out] stream  Buffer in which to skip the extension.
    //! \return Memory location after extension header.
    //----------------------------------------
    void skipRtpExtensionHeader(SharedBufferStream& stream) const;

    //========================================
    //! \brief Parse the jpeg header in the RTP data.
    //!
    //! \param[in, out] stream     Buffer from which to parse the RTP header.
    //! \param[out] rtpJpegHeader  Parsed RTP jpeg header.
    //----------------------------------------
    void parseRtpJpegHeader(SharedBufferStream& stream, RtpJpegHeader& rtpJpegHeader) const;

    //========================================
    //! \brief Parse the jpeg restart interval header in the RTP data.
    //!
    //! \param[in, out] stream         Buffer from which to parse the RTP restart interval header.
    //! \param[out]     rtpJpegHeader  Parsed RTP jpeg restart intervalheader.
    //----------------------------------------
    void parseRtpJpegRestartIntervalHeader(SharedBufferStream& stream, RtpJpegHeader& rtpJpegHeader) const;

    //========================================
    //! \brief Parse the jpeg quantization tables in the RTP data.
    //!
    //! \param[in, out] stream           Buffer from which to parse the RTP jpeg quantization tables.
    //! \param[out]     lumaTableData    Luma data bytes; \c nullptr if not found.
    //! \param[out]     chromaTableData  Chroma data bytes; \c nullptr if not found.
    //----------------------------------------
    void parseQuantizationTables(SharedBufferStream& stream,
                                 std::shared_ptr<std::vector<char>>& lumaTableData,
                                 std::shared_ptr<std::vector<char>>& chromaTableData) const;

private:
    CameraInfo m_cameraInfo{}; //!< Current camera info found in the RTP packet.

    std::vector<uint8_t> m_jpegHeader; //!< Current jpeg header found in the RTP packet.

    uint32_t m_currentRtpTimestamp; //!< Current rtp timestamp.

    uint32_t
        m_currentImageTimestamp; //!< Current rtp timestamp of image. Stored from package containing the start of the image,
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
