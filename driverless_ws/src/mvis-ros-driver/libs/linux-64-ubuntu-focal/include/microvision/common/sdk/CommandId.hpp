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
//! \date Jan 12, 2015
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/Unconvertable.hpp>
#include <microvision/common/sdk/misc/ToHex.hpp>

#include <microvision/common/sdk/io.hpp>
#include <microvision/common/sdk/bufferIO.hpp>

#include <boost/functional/hash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandId final : public microvision::common::sdk::ComparableUnconvertable<uint16_t>
{
public:
    enum class Id : uint16_t
    {
        // ECU commands
        CmdManagerSetFilter     = 0x0005, ///< Sets the data type filter, EMPTY REPLY
        CmdManagerAppBaseCtrl   = 0x000B, //!< EMPTY REPLY
        CmdManagerAppBaseStatus = 0x000C, //!<

        // LUX3 commands
        CmdLuxReset                  = 0x0000, //!< ID of the Reset command, NO REPLY!
        CmdLuxGetStatus              = 0x0001, //!< ID of the GetStatus command
        CmdLuxSaveConfig             = 0x0004, //!< ID of the SaveConfig command, EMPTY REPLY!
        CmdLuxSetParameter           = 0x0010, //!< sets a parameter in the sensor, EMPTY REPLY!
        CmdLuxGetParameter           = 0x0011, //!< reads a parameter from the sensor
        CmdLuxResetDefaultParameters = 0x001A, //!< resets all parameters to the factory defaults, EMPTY REPLY!
        CmdLuxStartMeasure           = 0x0020, //!< starts the measurement with the currently configured settings
        CmdLuxStopMeasure            = 0x0021, //!< stops the measurement
        CmdLuxSetNtpTimestampSync    = 0x0034 //!< set the complete NtpTime stamp, EMPTY REPLY!
    };

public:
    explicit CommandId(const uint16_t cId) : microvision::common::sdk::ComparableUnconvertable<uint16_t>(cId) {}
    CommandId(const Id c) : microvision::common::sdk::ComparableUnconvertable<uint16_t>(uint16_t(c)) {}

    operator Id() { return static_cast<Id>(m_data); }

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

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, TT& value);
    template<typename TT>
    friend void readLE(std::istream& is, TT& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const TT& value);
    template<typename TT>
    friend void writeLE(std::ostream& os, const TT& value);

}; // CommandId

//==============================================================================

//==============================================================================

template<>
inline std::string toHex<CommandId>(const CommandId c)
{
    return toHex(uint16_t(c));
}

//==============================================================================
template<>
inline void readBE<CommandId>(std::istream& is, CommandId& c)
{
    microvision::common::sdk::readBE(is, c.m_data);
}

//==============================================================================
template<>
inline void readLE<CommandId>(std::istream& is, CommandId& c)
{
    microvision::common::sdk::readLE(is, c.m_data);
}

//==============================================================================
template<>
inline void writeBE<CommandId>(std::ostream& os, const CommandId& c)
{
    microvision::common::sdk::writeBE(os, c.m_data);
}

//==============================================================================
template<>
inline void writeLE<CommandId>(std::ostream& os, const CommandId& c)
{
    microvision::common::sdk::writeLE(os, c.m_data);
}

//==============================================================================

std::ostream& operator<<(std::ostream& os, const CommandId& cmdId);
std::ostream& operator<<(std::ostream& os, const CommandId::Id& cmdId);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
namespace boost {
template<>
struct hash<microvision::common::sdk::CommandId>
{
    std::size_t operator()(microvision::common::sdk::CommandId const& cId) const
    {
        return boost::hash_value(uint16_t(cId));
    }
}; // hash<microvision::common::sdk::CommandId>
} // namespace boost
//==============================================================================
