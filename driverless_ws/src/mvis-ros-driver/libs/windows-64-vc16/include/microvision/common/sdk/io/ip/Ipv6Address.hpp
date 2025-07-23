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

#include <microvision/common/sdk/io/ip/IpAddress.hpp>

#include <microvision/common/logging/logging.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//=================================================
//! \brief Pre definition of class Ipv6Address.
//-------------------------------------------------
class Ipv6Address;

//=================================================
//! \brief Nullable Ipv6Address pointer
//-------------------------------------------------
using Ipv6AddressPtr = std::shared_ptr<Ipv6Address>;

//==============================================================================
//!\brief Implementation to represent IP version 6 address functionality.
//!
//! \extends IpAddress
//!
//! \note Use ::makeIpv6 to create a new specific IP v6 address object.
//------------------------------------------------------------------------------
class Ipv6Address final : public IpAddress
{
public:
    friend Ipv6AddressPtr makeIpv6(const std::string& ipString);

    //========================================
    //! \brief Type of bytes array with max size of ipv6.
    //----------------------------------------
    using BytesType = std::array<uint8_t, 16>;

private:
    //========================================
    //! \brief Logger
    //----------------------------------------
    static logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor creates unspecific (any) IP address.
    //----------------------------------------
    Ipv6Address();

    //========================================
    //! \brief Constructor creates IP address from bytes and optional scope id.
    //! \param[in] bytes    Byte array of IP address
    //! \param[in] scopeId  Optional ipv6 scope id (default: 0)
    //----------------------------------------
    Ipv6Address(const BytesType& bytes, const uint64_t scopeId = 0);

    //========================================
    //! \brief Constructor creates copy of IP address.
    //! \param[in] other  Ip address to copy
    //----------------------------------------
    Ipv6Address(const Ipv6Address& other);

    //========================================
    //! \brief Constructor to move IP address.
    //! \param[in, out] other  Ip address to move
    //----------------------------------------
    Ipv6Address(Ipv6Address&& other);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Ipv6Address() override;

public:
    //========================================
    //! \brief Get byte array representation of IP address.
    //! \retuns Byte array of IP address
    //----------------------------------------
    const BytesType& getBytes() const;

    //========================================
    //! \brief Get scope id of ipv6
    //! \retuns Scope id
    //----------------------------------------
    uint64_t getScopeId() const;

    //========================================
    //! \brief Assign bytes array to change IP address.
    //! \param[in] bytes  Byte array of IP address
    //! \retuns Reference to this
    //----------------------------------------
    Ipv6Address& operator=(const BytesType& bytes);

    //========================================
    //! \brief Assign scope id to change IP address.
    //! \param[in] scopeId  Scope id of IP address
    //! \retuns Reference to this
    //----------------------------------------
    Ipv6Address& operator=(const uint64_t scopeId);

    //========================================
    //! \brief Assign other IP address to become a copy
    //! \param[in] other  Ip address version 6
    //! \retuns Reference to this
    //----------------------------------------
    Ipv6Address& operator=(const Ipv6Address& other);

    //========================================
    //! \brief Assign other IP address to move values
    //! \param[in, out] other  Ip address version 6
    //! \retuns Reference to this
    //----------------------------------------
    Ipv6Address& operator=(Ipv6Address&& other);

    //========================================
    //! \brief Compare IP addresses on equality.
    //! \param[in] lhs  Ip Address on the left side of the operator.
    //! \param[in] rhs  Ip Address on the right side of the operator.
    //! \returns Either \c true if IP addresses equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const Ipv6Address& lhs, const Ipv6Address& rhs);

    //========================================
    //! \brief Compare IP addresses on inequality.
    //! \param[in] lhs  Ip Address on the left side of the operator.
    //! \param[in] rhs  Ip Address on the right side of the operator.
    //! \returns Either \c true if IP addresses unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const Ipv6Address& lhs, const Ipv6Address& rhs);

public:
    //========================================
    //! \brief Get IP version
    //! \return Ip version
    //----------------------------------------
    IpVersion getVersion() const override;

    //========================================
    //! \brief Checks whether IP address is unspecific.
    //! \return Either \c true if IP address is unspecific or otherwise \c false.
    //----------------------------------------
    bool isAny() const override;

    //========================================
    //! \brief Checks whether IP address is kind of multicast address.
    //! \return Either \c true if IP address is a kind of multicast address or otherwise \c false.
    //----------------------------------------
    bool isMulticast() const override;

    //========================================
    //! \brief Get string representation of IP address.
    //! \return The typical string representation of IP address.
    //----------------------------------------
    std::string toString() const override;

private:
    //========================================
    //! \brief Byte array of ipv6
    //----------------------------------------
    BytesType m_value;

    //========================================
    //! \brief Scope id of ipv6
    //----------------------------------------
    uint64_t m_scope;
};

//=================================================
//! \brief Create an address from an IPv6 address in hexadecimal notation.
//! The supported format of the string is <ipv6_address>%<scope_id>, where the part
//! after <ipv6_address> is optional.
//! \param[in] ipString  String representation of IP address.
//! \returns Pointer to ipv6 address instance or \c nullptr if string is empty.
//!
//! \note In most use cases when creating an IP address object you should prefer a general IpAddress created by ::makeIp.
//-------------------------------------------------
Ipv6AddressPtr makeIpv6(const std::string& ipString);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
