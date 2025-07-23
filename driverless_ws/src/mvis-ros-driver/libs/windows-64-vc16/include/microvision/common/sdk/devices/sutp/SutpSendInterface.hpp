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
//! \date May 3, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>
#include <microvision/common/logging/logging.hpp>

#include <boost/function.hpp>
#include <boost/asio.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class SutpSendInterface final
{
public:
    using SendingFunction = boost::function<bool(boost::asio::streambuf& outBuff)>;

public:
    SutpSendInterface(SendingFunction sender);
    virtual ~SutpSendInterface();

public:
    void send(const IdcDataHeader& dh, const char* const buffer);
    //	void send(const IdcDataHeader& dh, std::istream& is);

protected:
    void serializeSutpHeader(std::ostream& os,
                             const uint16_t fragmentId,
                             const uint16_t nbOfFragments,
                             const DataTypeId dtId);

protected:
    static const uint64_t timeOutDurationMs;
    static const int64_t fragmentSize;

    static constexpr const char* loggerId = "microvision::common::sdk::SutpSendInterface";
    static microvision::common::logging::LoggerSPtr logger;

protected:
    SendingFunction m_sending;

    //! sequenceNumber counter for sending SUTP packages
    uint16_t m_segNo;

    //! id of the device that owns this connection
    uint8_t m_deviceId;

    uint32_t m_nbOfSendPackages;

}; // SutpSendInterface

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
