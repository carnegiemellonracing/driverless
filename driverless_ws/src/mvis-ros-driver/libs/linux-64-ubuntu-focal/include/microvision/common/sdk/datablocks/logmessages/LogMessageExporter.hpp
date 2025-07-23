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
//! \date May 28, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/SdkExceptions.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<class C>
class LogMessageExporter
{
public:
    std::streamsize getSerializedSize(const C& container) const;
    bool serialize(std::ostream& os, const C& container) const;
}; // LogMessageExporter

//==============================================================================

template<class C>
std::streamsize LogMessageExporter<C>::getSerializedSize(const C& container) const
{
    return static_cast<std::streamsize>(1 + container.m_message.size());
}

//==============================================================================

template<class C>
bool LogMessageExporter<C>::serialize(std::ostream& os, const C& container) const
{
    const int64_t startPos = streamposToInt64(os.tellp());

    microvision::common::sdk::writeBE(os, static_cast<uint8_t>(container.m_msgTraceLevel));
    os.write(container.m_message.c_str(), std::streamsize(container.m_message.size()));

    return !os.fail() && ((streamposToInt64(os.tellp()) - startPos) == this->getSerializedSize(container));
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
