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
//! \date Dec 4, 2020
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//========================================
//! \brief Supported Ip version
//----------------------------------------
enum class IpVersion : uint8_t
{
    IPv4 = 0, //!< Internet protocol version 4
    IPv6 = 1 //!< Internet protocol version 6
};

//==============================================================================
//!\brief Class representing the basic IP address functionality used by
//! configuration of network connections.
//!
//! \sa NetworkConfiguration
//!
//! \note Use ::makeIp to create a new IP address object.
//------------------------------------------------------------------------------
class IpAddress
{
public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~IpAddress();

public:
    //========================================
    //! \brief Compare IP addresses on equality.
    //! \param[in] lhs  Ip Address on the left side of the operator.
    //! \param[in] rhs  Ip Address on the right side of the operator.
    //! \returns Either \c true if IP addresses equals or otherwise \c false.
    //! \note Does not compare ipv6 compatible ipv4 address with ipv4 address.
    //----------------------------------------
    friend bool operator==(const IpAddress& lhs, const IpAddress& rhs);

    //========================================
    //! \brief Compare IP addresses on inequality.
    //! \param[in] lhs  Ip Address on the left side of the operator.
    //! \param[in] rhs  Ip Address on the right side of the operator.
    //! \returns Either \c true if IP addresses unequals or otherwise \c false.
    //! \note Does not compare ipv6 compatible ipv4 address with ipv4 address.
    //----------------------------------------
    friend bool operator!=(const IpAddress& lhs, const IpAddress& rhs);

public:
    //========================================
    //! \brief Get IP version
    //! \return Ip version
    //----------------------------------------
    virtual IpVersion getVersion() const = 0;

    //========================================
    //! \brief Checks whether IP address is unspecific.
    //! \return Either \c true if IP address is unspecific or otherwise \c false.
    //----------------------------------------
    virtual bool isAny() const = 0;

    //========================================
    //! \brief Checks whether IP address is kind of multicast address.
    //! \return Either \c true if IP address is a kind of multicast address or otherwise \c false.
    //----------------------------------------
    virtual bool isMulticast() const = 0;

    //========================================
    //! \brief Get string representation of IP address.
    //! \return The typical string representation of IP address.
    //----------------------------------------
    virtual std::string toString() const = 0;
};

//=================================================
//! \brief Nullable IpAddress pointer type
//-------------------------------------------------
using IpAddressPtr = std::shared_ptr<IpAddress>;

//=================================================
//! \brief Create an address from an IPv4 address string in dotted decimal form,
//!        or from an IPv6 address in hexadecimal notation.
//! \param[in] ipString  String representation of IP address.
//! \returns Pointer to IP address implementation instance
//!          or \c nullptr if string is empty or IP format/version does not supported.
//-------------------------------------------------
IpAddressPtr makeIp(const std::string& ipString);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
