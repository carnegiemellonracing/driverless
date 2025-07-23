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

#include <istream>
#include <ostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ScanTrailerIn2310 final
{
public: // public types
    enum class ConfigurationRegisterFlags : uint16_t
    {
        MirrorSide        = 0x0001U,
        ReducedApdVoltage = 0x0002U
    };

    enum class MirrorSide : uint16_t
    {
        MirrorSide0 = 0,
        MirrorSide1 = 1
    };

public: // static methods
    static std::streamsize getSerializedSize_static() { return 16; }

public: // constructor/destructor
    ScanTrailerIn2310();
    virtual ~ScanTrailerIn2310();

public: // operators
    //! Equality predicate
    bool operator==(const ScanTrailerIn2310& other) const;
    bool operator!=(const ScanTrailerIn2310& other) const;

public: // Snippet interface
    virtual std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public: // getter
    uint16_t getOperatingRegister() const { return m_operatingRegister; }
    uint16_t getWarning() const { return m_warning; }
    uint16_t getError() const { return m_error; }
    uint16_t getScanPointCounter() const { return m_scanPointCounter; }
    uint16_t getConfigurationRegister() const { return m_configurationRegister; }
    uint16_t getReservedTrailer7() const { return m_reservedTrailer7; }

public: // derived getter
    MirrorSide getMirrorSide() const
    {
        return ((m_configurationRegister & static_cast<uint16_t>(ConfigurationRegisterFlags::MirrorSide))
                == static_cast<uint16_t>(ConfigurationRegisterFlags::MirrorSide))
                   ? MirrorSide::MirrorSide1
                   : MirrorSide::MirrorSide0;
    }
    bool isApdVoltageReduced() const
    {
        return ((m_configurationRegister & static_cast<uint16_t>(ConfigurationRegisterFlags::ReducedApdVoltage))
                == static_cast<uint16_t>(ConfigurationRegisterFlags::ReducedApdVoltage));
    }

public: // setter
    void setOperatingRegister(const uint16_t reg) { m_operatingRegister = reg; }
    void setWarning(const uint16_t warning) { m_warning = warning; }
    void setError(const uint16_t err) { m_error = err; }
    void setScanPointCounter(const uint16_t cnt) { m_scanPointCounter = cnt; }
    void setConfigurationRegister(const uint16_t side) { m_configurationRegister = side; }

public: // derived setter
    void setMirrorSide(const MirrorSide mirrorSide)
    {
        modifyConfigRegisterFlag(mirrorSide == MirrorSide::MirrorSide1, ConfigurationRegisterFlags::MirrorSide);
    }
    void setApdVoltageReduced(const bool isReduced)
    {
        modifyConfigRegisterFlag(isReduced, ConfigurationRegisterFlags::ReducedApdVoltage);
    }

protected:
    void modifyConfigRegisterFlag(const bool flagValue, const ConfigurationRegisterFlags bit)
    {
        if (flagValue)
        {
            m_configurationRegister = uint16_t(m_configurationRegister | static_cast<uint16_t>(bit));
        }
        else
        {
            m_configurationRegister = uint16_t(m_configurationRegister & ~static_cast<uint16_t>(bit));
        }
    }

public: // public static const attributes
    static const uint16_t blockId;

protected: // protected attributes
    uint16_t m_operatingRegister;
    uint16_t m_warning;
    uint16_t m_error;
    uint16_t m_reservedTrailer4;
    uint16_t m_scanPointCounter;
    uint16_t m_configurationRegister;
    uint16_t m_reservedTrailer7;
}; // ScanTrailerIn2310

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
