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
//! \date Apr 10, 2015
//------------------------------------------------------------------------------
#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io.hpp>
#include <microvision/common/sdk/bufferIO.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ParameterData final
{
public:
    explicit ParameterData(const uint32_t data) : m_data(data) {}
    explicit ParameterData(const int32_t data) : m_data(uint32_t(data)) {}

public:
    operator uint32_t() const { return m_data; }

public: // BE io
    std::istream& readBE(std::istream& is)
    {
        microvision::common::sdk::readBE(is, this->m_data);
        return is;
    }

    std::ostream& writeBE(std::ostream& os) const
    {
        microvision::common::sdk::writeBE(os, this->m_data);
        return os;
    }

    void readBE(const char*& target) { microvision::common::sdk::readBE(target, this->m_data); }
    void writeBE(char*& target) const { microvision::common::sdk::writeBE(target, this->m_data); }

public: // LE io
    std::istream& readLE(std::istream& is)
    {
        microvision::common::sdk::readLE(is, this->m_data);
        return is;
    }

    std::ostream& writeLE(std::ostream& os) const
    {
        microvision::common::sdk::writeLE(os, this->m_data);
        return os;
    }

    void readLE(const char*& target) { microvision::common::sdk::readLE(target, this->m_data); }
    void writeLE(char*& target) const { microvision::common::sdk::writeLE(target, this->m_data); }

protected:
    uint32_t m_data;
}; // ParameterData

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
