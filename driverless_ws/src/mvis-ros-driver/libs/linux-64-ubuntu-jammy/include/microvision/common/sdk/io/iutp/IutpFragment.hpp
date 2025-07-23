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
//! \date Sep 30, 2019
//!
//! Structures for the IUTP protocol
//!
//! The IUTP is a small and scalable transport protocol on top of UDP.
//!
//! In its current version it allows to detect messages received out of order and missing fragments
//! The IUTP is not reliable, thus it cannot assure a transmission was successfully received.
//!
//! The protocol allows to transfer messages of a maximum payload size of approximately 4GB.
//!
//! The transfer of large messages is realized by fragmenting the payload in multiple frames. Messages up to
//! MAX_UDP_DATAGRAM_SIZE-(sizeof(InitialFragmentHeader) (roughly 65kB) can be transferred within one UDP
//! datagram that is fragmented by the IP layer. But depending on the implementation smaller fragments are also
//! possible
//!
//! The protocol uses two different headers InitialFragmentHeader and FollowupFragmentHeader that can be identified by
//! their protocol type within the first byte of the header. Each message starts with an InitialFragmentHeader datagram
//! containing the InitialFragmentHeader header followed by the payload. If the payload does not fit in the datagram
//! one or more FollowupFragments are sent starting with the FollowupFragmentHeader header followed by the payload.
//! The size of the payload can be determined by the UDP datagram length minus the header size.
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace iutp {
//==============================================================================

//========================================
//! Enum defining the fragment type
//----------------------------------------
enum class FragmentType : uint8_t
{
    InitialFragment   = 0x01U, //!< Type of InitialFragment
    FollowingFragment = 0x02U, //!< Type of FollowupFragment
};

const uint8_t version{0x01U}; //!< Version of this protocol definition
const uint8_t fragmentTypeBitMask{0xF0U}; //!< Bit mask to set the version zero.
const uint8_t fragmentVersionBitMask{0x0FU}; //!< Bit mask to set the type zero.
const uint8_t fragmentTypeBitShift{4U}; //!< Bit shift to get or set the type.

const uint16_t indexToLengthOffset{0x1U}; //!< Offset between index and length.

const size_t initialFragmentHeaderSize{12U};
const size_t followupFragmentHeaderSize{14U};

//==============================================================================
} // namespace iutp
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
