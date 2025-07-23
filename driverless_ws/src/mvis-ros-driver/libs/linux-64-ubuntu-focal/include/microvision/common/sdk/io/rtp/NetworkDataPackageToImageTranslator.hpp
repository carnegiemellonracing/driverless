//==============================================================================
//!\file
//!
//! \brief Definition of translator to image from package stream containing RTP data.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 28, 2022
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/rtp/RtpParser.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2403.hpp>

#include <microvision/common/sdk/misc/SharedBufferStream.hpp>
#include <microvision/common/sdk/io/NetworkDataPackage.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate to IdcDataPackage containing image from package stream containing RTP data.
//! \extends DataStreamTranslator<NetworkDataPackage, IdcDataPackage>
//------------------------------------------------------------------------------
class NetworkDataPackageToImageTranslator final : public DataStreamTranslator<NetworkDataPackage, IdcDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<NetworkDataPackage, IdcDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::NetworkDataPackageToImageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    NetworkDataPackageToImageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    NetworkDataPackageToImageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of NetworkDataPackageToImageTranslator.
    //----------------------------------------
    NetworkDataPackageToImageTranslator(const NetworkDataPackageToImageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    NetworkDataPackageToImageTranslator(NetworkDataPackageToImageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~NetworkDataPackageToImageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of NetworkDataPackageToImageTranslator.
    //----------------------------------------
    NetworkDataPackageToImageTranslator& operator=(const NetworkDataPackageToImageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    NetworkDataPackageToImageTranslator& operator=(NetworkDataPackageToImageTranslator&&) = delete;

public:
    //========================================
    //! \brief Set device id for resulting images.
    //! \param[in] Device id to be set.
    //----------------------------------------
    void setDeviceId(const uint8_t deviceId);

public: // implements DataStreamTranslator<DataPackage, IdcDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate DataPackage to IdcDataPackage.
    //! \param[in] dataPackage  Input DataPackage containing RTP data to process to image.
    //! \returns Either \c true if input is valid, otherwise \c false.
    //----------------------------------------
    bool translate(const InputPtr& dataPackage) override;

    //========================================
    //! \brief Clean up the translator state.
    //! \note Call it to get memory free,
    //!       only if no more packages are coming in.
    //----------------------------------------
    void clear() override;

private:
    //========================================
    //! \brief Collect complete images from chunks contained in different packets.
    //! \param[in] imagechunk  Payload of one rtp packet.
    //----------------------------------------
    void processImageChunk(const std::shared_ptr<RtpParser::RtpPayload>& imagechunk);

    //========================================
    //! \brief Calculate the current RTP stream timestamp.
    //!
    //! Timestamps from RTP image packages only contain a tick which can be used to get the correct time with the help of a previously received sender report header timestamp.
    //! \returns Last timestamp of stream.
    //----------------------------------------
    NtpTime calculateTimestampFromRtpStream() const;

private:
    std::queue<RtpParser::RtpPayloadPtr>
        m_payloadQueue; //!< For correct times incoming payloads have be stored until sync arrived.

    std::vector<unsigned char> m_imagePartBuffer; //!< Buffer to collect image parts from different rtp packets.
    NtpTime m_timestamp; //!< Timestamp of first image chunk used as timestamp for image.

    RtpParser m_rtpParser; //!< Rtp parser to extract image payload chunks from rtp data packages.

    bool m_rtcpSyncMessageArrived; //!< Flag for waiting until a rtp sender report header is found.
    std::pair<uint32_t, NtpTime> m_rtpNtpTimeRecord; //!< Last timestamp/time pair received from rtp sender report.

    //========================================
    //! \brief Post processing function set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

    //========================================
    //! \brief Device id to be set on image data.
    //----------------------------------------
    uint8_t m_deviceId;
}; // class NetworkDataPackageToImageTranslator

//==============================================================================
//! \brief Nullable NetworkDataPackageToImageTranslator pointer.
//------------------------------------------------------------------------------
using DataPackageToImageTranslatorPtr = std::shared_ptr<NetworkDataPackageToImageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
