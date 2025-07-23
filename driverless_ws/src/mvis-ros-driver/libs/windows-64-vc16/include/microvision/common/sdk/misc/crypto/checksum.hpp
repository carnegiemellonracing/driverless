//==============================================================================
//! \file
//!
//! \brief Helper for checkum calculation.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 04, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <boost/crc.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace crypto {
//==============================================================================

//==============================================================================
//! \brief Polynomial is 0xF4ACFB13 defined in the AUTOSAR (30 November 2020), AUTOSAR Classic Platform release R20-11, Specification of CRC Routines. This Polynomial is IP of AUTOSAR.
//!
//! Initial value: 0xFFFFFFFF
//! Final xor:     0xFFFFFFFF
//!
//! Input data and CRC are reflected (bits swapped).
//! The computed CRC is complemented.
//------------------------------------------------------------------------------
using AutosarCrc32 = boost::crc_optimal<32, 0xF4ACFB13, 0xFFFFFFFF, 0xFFFFFFFF, true, true>;

//==============================================================================
} // namespace crypto
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
