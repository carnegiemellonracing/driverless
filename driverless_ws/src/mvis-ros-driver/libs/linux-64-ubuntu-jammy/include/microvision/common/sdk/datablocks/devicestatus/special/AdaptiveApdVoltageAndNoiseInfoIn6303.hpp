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
//! \date Jan 29, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6303.hpp>
#include <microvision/common/sdk/Math.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class AdaptiveApdVoltageAndNoiseInfoIn6303 final : public DeviceStatus6303::UserDefinedStructBase
{
public:
    constexpr static const std::size_t nbOfSectors{10};
    constexpr static const std::size_t nbOfReserved{8};

public:
    //========================================

    class ApdSector
    {
    public:
        ApdSector()          = default;
        virtual ~ApdSector() = default;

    public: // getter
        float getAdaptiveApdVoltage() const { return m_adaptiveApdVoltage; }
        float getReducedApdOffset() const { return m_reducedApdOffset; }
        uint16_t getNoise() const { return m_noise; }

    public: // setter
        void setAdaptiveApdVoltage(float adaptiveApdVoltage) { m_adaptiveApdVoltage = adaptiveApdVoltage; }
        void setReducedApdOffset(float reducedApdOffset) { m_reducedApdOffset = reducedApdOffset; }
        void setNoise(uint16_t noise) { m_noise = noise; }

    public:
        static uint8_t getSerializedSizeStatic();
        bool serialize(std::ostream& os) const;
        bool deserialize(std::istream& is);

    private:
        float m_adaptiveApdVoltage{NaN};
        float m_reducedApdOffset{NaN};
        uint16_t m_noise{std::numeric_limits<uint16_t>::max()};
    }; // ApdSector

    //========================================

    using ApdSectorArray = std::array<ApdSector, nbOfSectors>;

    //========================================

public:
    //========================================

    enum class ApdVoltageFlags : uint8_t
    {
        Reduced   = 0x01U,
        CalcError = 0x02U,
        Reserved2 = 0x04U,
        Reserved3 = 0x08U,
        Reserved4 = 0x10U,
        Reserved5 = 0x20U,
        Reserved6 = 0x40U,
        Reserved7 = 0x80U
    };

    //========================================

public:
    AdaptiveApdVoltageAndNoiseInfoIn6303()
      : DeviceStatus6303::UserDefinedStructBase(DeviceStatus6303::ContentId::AdaptiveApdVoltageNoiseArray)
    {}
    virtual ~AdaptiveApdVoltageAndNoiseInfoIn6303() = default;

public: // getter
    uint32_t getProcessingUnitAndVersion() const { return m_procUnitAndVersion; }
    ApdSectorArray& getSectors() { return m_sectors; }
    const ApdSectorArray& getSectors() const { return m_sectors; }
    uint16_t getScanNumber() const { return m_scanNumber; }
    uint8_t getFlags() const { return m_flags; }
    const std::array<uint8_t, 8>& getReserved() const { return m_reserved; }

public: // setter
    void setProcUnitAndVersion(uint32_t procUnitAndVersion) { m_procUnitAndVersion = procUnitAndVersion; }
    void setSectors(const ApdSectorArray& sectors) { m_sectors = sectors; }
    void setScanNumber(uint16_t scanNumber) { m_scanNumber = scanNumber; }
    void setFlags(uint8_t flags) { m_flags = flags; }
    void setReserved(const std::array<uint8_t, 8>& reserved) { m_reserved = reserved; }

public:
    static uint8_t getSerializedSizeStatic();
    uint8_t getSerializedSize() const override { return getSerializedSizeStatic(); }
    bool serialize(char*& buf) const override;
    bool deserialize(const DeviceStatus6303::ContentDescr& cd) override;

private:
    //========================================
    //! \brief Processing unit that estimated and filled the data.
    //----------------------------------------
    ///
    uint32_t m_procUnitAndVersion{0};

    uint16_t m_scanNumber{std::numeric_limits<uint16_t>::max()};

    //========================================
    //! \brief Sectors containing apd values. Sectors are ordered in scanning direction (from left to right, i.e.
    //! from positive to negative angle ticks).
    //----------------------------------------
    ApdSectorArray m_sectors{};

    uint8_t m_flags{0};

    std::array<uint8_t, nbOfReserved> m_reserved{};
}; // AdaptiveApdVoltageAndNoiseInfoIn6303

//==============================================================================
//==============================================================================
//==============================================================================

bool operator==(const AdaptiveApdVoltageAndNoiseInfoIn6303& lhs, const AdaptiveApdVoltageAndNoiseInfoIn6303& rhs);

//==============================================================================

inline bool operator!=(const AdaptiveApdVoltageAndNoiseInfoIn6303& lhs, const AdaptiveApdVoltageAndNoiseInfoIn6303& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================

bool operator==(const AdaptiveApdVoltageAndNoiseInfoIn6303::ApdSector& lhs,
                const AdaptiveApdVoltageAndNoiseInfoIn6303::ApdSector& rhs);

//==============================================================================

inline bool operator!=(const AdaptiveApdVoltageAndNoiseInfoIn6303::ApdSector& lhs,
                       const AdaptiveApdVoltageAndNoiseInfoIn6303::ApdSector& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
