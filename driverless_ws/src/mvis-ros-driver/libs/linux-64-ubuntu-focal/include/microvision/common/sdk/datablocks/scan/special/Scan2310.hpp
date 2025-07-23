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
//! \date Jan 11, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/special/ScanHeaderIn2310.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanInfoIn2310.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanPointIn2310.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanSegInfoIn2310.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanPointRefScanIn2310.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanPointDiagPulseIn2310.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanTrailerIn2310.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/scan/Scan.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Scala for HIL
//!
//! General data type: \ref microvision::common::sdk::Scan
//------------------------------------------------------------------------------
class Scan2310 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.scalafpgarawdata2310"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    Scan2310();
    virtual ~Scan2310();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // DataBlock interface
    uint32_t getMagicWord() const { return magicWord; }
    uint16_t getInternalDataTypeId() const { return internalDataTypeId; }
    uint16_t getReserved00() const { return m_reserved00; }
    uint32_t getPayLoadSize() const { return m_payloadSize; }

    uint32_t getUtcSeconds() const { return m_utcSeconds; }
    uint32_t getFlexrayMeasTime() const { return m_flexrayMeasTime; }
    uint16_t getReserved01() const { return m_reserved01; }
    uint8_t getReserved02() const { return m_reserved02; }
    uint8_t getFlexrayMasterClock() const { return m_flexrayMasterClock; }
    uint32_t getReserved03() const { return m_reserved03; }
    uint16_t getVersionInfo0() const { return m_versionInfo0; }
    uint16_t getVersionInfo1() const { return m_versionInfo1; }
    const NtpTime& getScanStartTime() const { return m_scanStartTime; }
    const NtpTime& getScanEndTime() const { return m_scanEndTime; }
    const NtpTime& getScanMidTime() const { return m_scanMidTime; }
    uint32_t getReserved04() const { return m_reserved04; }

    const ScanHeaderIn2310& getHeader() const { return m_header; }
    const ScanInfoIn2310& getInfo() const { return m_info; }
    const std::vector<ScanPointIn2310>& getPoints() const { return m_points; }
    std::vector<ScanPointIn2310>& getPoints() { return m_points; }
    const std::vector<ScanSegInfoIn2310>& getSegInfos() const { return m_segInfos; }
    std::vector<ScanSegInfoIn2310>& getSegInfos() { return m_segInfos; }
    const std::vector<ScanPointDiagPulseIn2310>& getDiagPulses() const { return m_diagPulses; }
    std::vector<ScanPointDiagPulseIn2310>& getDiagPulses() { return m_diagPulses; }
    const std::vector<ScanPointRefScanIn2310>& getRefScans() const { return m_refScans; }
    std::vector<ScanPointRefScanIn2310>& getRefScans() { return m_refScans; }
    const ScanTrailerIn2310& getTrailer() const { return m_trailer; }
    uint32_t getCrc32() const { return m_crc32; }

    uint32_t getReserved05() const { return m_reserved05; }
    uint64_t getReserved06() const { return m_reserved06; }

public:
    //void setReserved00(const uint16_t res) { m_reserved00 = res; }
    void setPayLoadSize(const uint32_t plSz) { m_payloadSize = plSz; }

    void setUtcSeconds(const uint32_t secs) { m_utcSeconds = secs; }
    void setFlexrayMeasTime(const uint32_t frMt) { m_flexrayMeasTime = frMt; }
    //void setReserved01(const uint16_t r01) { m_reserved01 = r01; }
    //void setReserved02(const uint8_t r02) { m_reserved02 = r02; }
    void setFlexrayMasterClock(const uint8_t frMc) { m_flexrayMasterClock = frMc; }
    //void setReserved03(const uint32_t r03) { m_reserved03 = r03; }
    void setVersionInfo0(const uint16_t vi0) { m_versionInfo0 = vi0; }
    void setVersionInfo1(const uint16_t vi1) { m_versionInfo1 = vi1; }
    void setScanStartTime(const NtpTime& startTime) { m_scanStartTime = startTime; }
    void setScanEndTime(const NtpTime& endTime) { m_scanEndTime = endTime; }
    void setScanMidTime(const NtpTime& midTime) { m_scanMidTime = midTime; }
    //void setReserved04(const uint32_t r04) { m_reserved04 = r04; }

    void setHeader(const ScanHeaderIn2310& header) { m_header = header; }
    void setInfo(const ScanInfoIn2310& info) { m_info = info; }
    void setTrailer(const ScanTrailerIn2310& trailer) { m_trailer = trailer; }
    void setCrc32(const uint32_t crc) { m_crc32 = crc; }

protected:
    // header
    static constexpr uint32_t magicWord{0x5CA7ADA7U}; // not const, will be read
    static constexpr uint16_t internalDataTypeId{0xD0D2U}; // not const, will be read
    uint16_t m_reserved00{0};
    uint32_t m_payloadSize{48};

    //payload
    uint32_t m_utcSeconds{0};
    uint32_t m_flexrayMeasTime{0};
    uint16_t m_reserved01{0};
    uint8_t m_reserved02{0};
    uint8_t m_flexrayMasterClock{0};
    uint32_t m_reserved03{0};
    uint16_t m_versionInfo0{0};
    uint16_t m_versionInfo1{0};
    NtpTime m_scanStartTime{};
    NtpTime m_scanEndTime{};
    NtpTime m_scanMidTime{};
    uint32_t m_reserved04{0};
    ScanHeaderIn2310 m_header{};
    ScanInfoIn2310 m_info{};
    std::vector<ScanPointIn2310> m_points{};
    std::vector<ScanSegInfoIn2310> m_segInfos{};
    std::vector<ScanPointDiagPulseIn2310> m_diagPulses{};
    std::vector<ScanPointRefScanIn2310> m_refScans{};
    ScanTrailerIn2310 m_trailer{};
    uint32_t m_crc32{0};
    uint32_t m_reserved05{0};
    uint64_t m_reserved06{0};
}; // Scan2310Container

//==============================================================================

bool operator==(const Scan2310& lhs, const Scan2310& rhs);
bool operator!=(const Scan2310& lhs, const Scan2310& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
