//==============================================================================
//! \file
//!
//! \brief Boost helpers.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 28, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <boost/asio.hpp>
#include <microvision/common/sdk/misc/Utils.hpp>
#include <microvision/common/sdk/io/ip/Ipv4Address.hpp>
#include <microvision/common/sdk/io/ip/Ipv6Address.hpp>

//==============================================================================

#if BOOST_ASIO_VERSION >= 101400
using BoostIoContext            = boost::asio::io_context;
using BoostIpv4NumericValueType = boost::asio::ip::address_v4::uint_type;
#else
using BoostIoContext            = boost::asio::io_service;
using BoostIpv4NumericValueType = decltype(boost::asio::ip::address_v4{}.to_ulong());
#endif

using BoostIpv6NumericValueType = decltype(boost::asio::ip::address_v6{}.scope_id());

//==============================================================================

inline boost::asio::ip::address getBoostIpAddress(const microvision::common::sdk::IpAddressPtr& ipAddress)
{
    if (ipAddress)
    {
        if (ipAddress->getVersion() == microvision::common::sdk::IpVersion::IPv4)
        {
            const auto ipv4Address = std::dynamic_pointer_cast<microvision::common::sdk::Ipv4Address>(ipAddress);
            return boost::asio::ip::address_v4{
                microvision::common::sdk::staticCastIfNeeded<uint32_t, BoostIpv4NumericValueType>(
                    ipv4Address->getNumeric())};
        }
        else if (ipAddress->getVersion() == microvision::common::sdk::IpVersion::IPv6)
        {
            const auto ipv6Address = std::dynamic_pointer_cast<microvision::common::sdk::Ipv6Address>(ipAddress);
            return boost::asio::ip::address_v6{
                ipv6Address->getBytes(),
                microvision::common::sdk::staticCastIfNeeded<uint64_t, BoostIpv6NumericValueType>(
                    ipv6Address->getScopeId())};
        }
    }
    return boost::asio::ip::address{};
}

//==============================================================================

inline int mapBoostToStdErrorCode(const boost::system::error_code& error)
{
    if (error.value() == static_cast<int>(boost::asio::error::operation_aborted))
    {
        return static_cast<int>(std::errc::operation_canceled);
    }
    else if (error.value() == static_cast<int>(boost::asio::error::no_permission))
    {
        return static_cast<int>(std::errc::permission_denied);
    }
    else if (error.value() == static_cast<int>(boost::asio::error::no_memory))
    {
        return static_cast<int>(std::errc::not_enough_memory);
    }
    else if (error.value() == static_cast<int>(boost::asio::error::try_again))
    {
        return static_cast<int>(std::errc::resource_unavailable_try_again);
    }
    else if (error.value() == static_cast<int>(boost::asio::error::no_such_device))
    {
        return static_cast<int>(std::errc::no_such_device);
    }
    else
    {
        return static_cast<int>(error.value());
    }
}

inline std::exception_ptr makeExceptionByBoostError(const boost::system::error_code& error, const std::string& message)
{
    int errorCode{mapBoostToStdErrorCode(error)};

    if (errorCode != 0)
    {
        std::system_error ex{std::make_error_code(static_cast<std::errc>(errorCode)), message};
        return std::make_exception_ptr(ex);
    }

    return nullptr;
}

//==============================================================================
