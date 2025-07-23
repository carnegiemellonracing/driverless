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

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Implementation to represent IP version 4 address functionality.
//!
//! \extends IpAddress
//!
//! \note Use ::makeIpv4 to create a new specific IP v4 address object.
//------------------------------------------------------------------------------
class Ipv4Address final : public IpAddress
{
public:
    //========================================
    //! \brief Type of bytes array with max size of ipv4.
    //----------------------------------------
    using BytesType = std::array<uint8_t, 4>;

public:
    //========================================
    //! \brief Default constructor creates unspecific (any) IP address.
    //----------------------------------------
    Ipv4Address();

    //========================================
    //! \brief Constructor creates IP address from bytes.
    //! \param[in] bytes  Byte array of IP address
    //----------------------------------------
    Ipv4Address(const BytesType& bytes);

    //========================================
    //! \brief Constructor creates IP address from numeric.
    //! \param[in] numeric  Numeric value of IP address
    //----------------------------------------
    Ipv4Address(const uint32_t numeric);

    //========================================
    //! \brief Constructor creates copy of IP address.
    //! \param[in] other  Ip address to copy
    //----------------------------------------
    Ipv4Address(const Ipv4Address& other);

    //========================================
    //! \brief Constructor to move IP address.
    //! \param[in, out] other  Ip address to move
    //----------------------------------------
    Ipv4Address(Ipv4Address&& other);

    //========================================
    //! \brief Default destructor
    //----------------------------------------
    ~Ipv4Address() override;

public:
    //========================================
    //! \brief Get numeric representation of IP address
    //! \retuns Numeric value of IP address
    //----------------------------------------
    uint32_t getNumeric() const;

    //========================================
    //! \brief Get byte array representation of IP address.
    //! \retuns Byte array of IP address
    //----------------------------------------
    BytesType toBytes() const;

    //========================================
    //! \brief Assign bytes array to change IP address.
    //! \param[in] bytes  Byte array of IP address
    //! \retuns Reference to this
    //----------------------------------------
    Ipv4Address& operator=(const BytesType& bytes);

    //========================================
    //! \brief Assign numeric value to change IP address.
    //! \param[in] numeric  Numeric value of IP address
    //! \retuns Reference to this
    //----------------------------------------
    Ipv4Address& operator=(const uint32_t numeric);

    //========================================
    //! \brief Assign other IP address to become a copy
    //! \param[in] other  Ip address version 4
    //! \retuns Reference to this
    //----------------------------------------
    Ipv4Address& operator=(const Ipv4Address& other);

    //========================================
    //! \brief Assign other IP address to move values
    //! \param[in, out] other  Ip address version 4
    //! \retuns Reference to this
    //----------------------------------------
    Ipv4Address& operator=(Ipv4Address&& other);

    //========================================
    //! \brief Compare IP addresses on equality.
    //! \param[in] lhs  Ip Address on the left side of the operator.
    //! \param[in] rhs  Ip Address on the right side of the operator.
    //! \returns Either \c true if IP addresses are equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const Ipv4Address& lhs, const Ipv4Address& rhs);

    //========================================
    //! \brief Compare IP addresses on inequality.
    //! \param[in] lhs  Ip Address on the left side of the operator.
    //! \param[in] rhs  Ip Address on the right side of the operator.
    //! \returns Either \c true if IP addresses are unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const Ipv4Address& lhs, const Ipv4Address& rhs);

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
    //! \brief Numeric value of ipv4
    //----------------------------------------
    uint32_t m_value;
};

//=================================================
//! \brief Nullabe Ipv4Address pointer
//-------------------------------------------------
using Ipv4AddressPtr = std::shared_ptr<Ipv4Address>;

//=================================================
//! \brief Create an address from an IPv4 address string in dotted decimal notation.
//! \param[in] ipString  String representation of IP address.
//! \returns Pointer to ipv4 address instance or \c nullptr if string is empty.
//!
//! \note In most use cases when creating an IP address object you should prefer a general IpAddress created by ::makeIp.
//-------------------------------------------------
Ipv4AddressPtr makeIpv4(const std::string& ipString);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
