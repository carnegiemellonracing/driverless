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
//! \date Sep 17, 2015
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/special/ScanPointBaseIn2310.hpp>

#include <istream>
#include <ostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class MICROVISION_SDK_API ScanPointIn2310 final : public ScanPointBaseIn2310
{
public:
    ScanPointIn2310();
    virtual ~ScanPointIn2310();

public:
    //! Equality predicate
    bool operator==(const ScanPointIn2310& other) const;
    bool operator!=(const ScanPointIn2310& other) const;

public:
    virtual uint16_t getBlockId() const override { return blockId; }

public:
    static const uint16_t blockId;
}; // ScanPointIn2310

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
