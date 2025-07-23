//==============================================================================
//!
//! \file
//!
//! \brief Perception performance data snippet.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date May 7th, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Snippet used to hold perception performance info data received by LdeDevice.
//------------------------------------------------------------------------------
class PerceptionPerformanceInfo final
{
public:
    //========================================
    //! \brief Serialized size of an PerceptionPerformanceInfo.
    //----------------------------------------
    static constexpr uint8_t serializedSize{3};

public:
    //========================================
    //! \brief Enum specifying the detection range.
    //----------------------------------------
    enum class DetectionRange : uint8_t
    {
        Low    = 0,
        Medium = 1,
        High   = 2
    };

public:
    //========================================
    //! \brief Get the serialized size of an PerceptionPerformanceInfo.
    //! \return Return the serialized size of an PerceptionPerformanceInfo.
    //----------------------------------------
    static constexpr std::streamsize getSerializedSize_static() { return serializedSize; }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    PerceptionPerformanceInfo();

    //========================================
    //! \brief Constructor.
    //! \param[in] blockage        Blocked flag.
    //! \param[in] detectionRange  Detection range (estimated).
    //! \param[in] confidence      Confidence of the estimation.
    //----------------------------------------
    PerceptionPerformanceInfo(const bool blockage, const DetectionRange detectionRange, const uint8_t confidence);

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~PerceptionPerformanceInfo();

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    PerceptionPerformanceInfo(const PerceptionPerformanceInfo& other);

    //========================================
    //! \brief Comparison operator
    //----------------------------------------
    PerceptionPerformanceInfo& operator=(const PerceptionPerformanceInfo& other);

public:
    friend void readBE(std::istream& is, PerceptionPerformanceInfo& value);
    friend void writeBE(std::ostream& os, const PerceptionPerformanceInfo& value);

    //========================================
    //! \brief Get the serialized size of this PerceptionPerformanceInfo.
    //! \return Return the serialized size of this PerceptionPerformanceInfo.
    //----------------------------------------
    std::streamsize getSerializedSize() const { return serializedSize; }

public: // getter
    //========================================
    //! \brief Get the blockage flag.
    //! \return Return the blockage flag.
    //----------------------------------------
    bool getBlockage() const { return m_blockage; }

    //========================================
    //! \brief Get the estimated detection range.
    //! \return Return the detection range.
    //----------------------------------------
    DetectionRange getDetectionRange() const { return m_detectionRange; }

    //========================================
    //! \brief Get the confidence of the detection range estimation.
    //! \return Return the confidence value in percent.
    //----------------------------------------
    uint8_t getEstimationConfidence() const { return m_estimationConfidence; }

private:
    //========================================
    //! \brief Binary flag to indicate if the sensor is considered as blocked.
    //----------------------------------------
    bool m_blockage{false};

    //========================================
    //! \brief Estimated detection range.
    //----------------------------------------
    DetectionRange m_detectionRange{DetectionRange::Low};

    //========================================
    //! \brief Estimation confidence in %.
    //----------------------------------------
    uint8_t m_estimationConfidence{0};

}; // PerceptionPerformanceInfo

//==============================================================================

//========================================
//! \brief Deserialize an PerceptionPerformanceInfo from \a is.
//! \param[in, out] is     An input stream containing the serialized
//!                        data of the PerceptionPerformanceInfo to
//!                        be read.
//! \param[out]     value  PerceptionPerformanceInfo to be filled.
//----------------------------------------
void readBE(std::istream& is, microvision::common::sdk::PerceptionPerformanceInfo& value);

//========================================
//! \brief Serialize this PerceptionPerformanceInfo to \a os.
//! \param[in, out] os     An output stream this object shall be
//!                        written to.
//! \param[in]      value  Object that shall be written.
//----------------------------------------
void writeBE(std::ostream& os, const microvision::common::sdk::PerceptionPerformanceInfo& value);

//==============================================================================

//========================================
//! \brief Serialize this DetectionRange to \a os.
//! \param[in, out] os     An output stream this object shall be
//!                        written to.
//! \param[in]      value  Object that shall be written.
//----------------------------------------
template<>
inline void writeLE<PerceptionPerformanceInfo::DetectionRange>(std::ostream& os,
                                                               const PerceptionPerformanceInfo::DetectionRange& value)
{
    using EnumIntType = std::underlying_type<PerceptionPerformanceInfo::DetectionRange>::type;
    const EnumIntType tmp{static_cast<EnumIntType>(value)};
    writeLE(os, tmp);
}

//==============================================================================

//========================================
//! \brief Deserialize an DetectionRange from \a is.
//! \param[in, out] is     An input stream containing the serialized
//!                        data of the DetectionRange to
//!                        be read.
//! \param[out]     value  PerceptionPerformanceInfo to be filled.
//----------------------------------------
template<>
inline void readLE<PerceptionPerformanceInfo::DetectionRange>(std::istream& is,
                                                              PerceptionPerformanceInfo::DetectionRange& value)
{
    using EnumIntType = std::underlying_type<PerceptionPerformanceInfo::DetectionRange>::type;
    EnumIntType tmp;
    readLE(is, tmp);
    value = static_cast<PerceptionPerformanceInfo::DetectionRange>(tmp);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
