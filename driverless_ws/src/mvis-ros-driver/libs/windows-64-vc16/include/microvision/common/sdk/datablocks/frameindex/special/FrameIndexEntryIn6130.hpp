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
//! \date May 11, 2015
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/NtpTime.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class FrameIndexEntryIn6130 final
{
public:
    static std::streamsize getSerializedSize_static();

public:
    FrameIndexEntryIn6130() : m_filePosition{0}, m_timeOffsetMs{0}, m_deviceId{255} {}

    FrameIndexEntryIn6130(const uint64_t filePos, const uint64_t timeOffsetMs, const uint8_t deviceId)
      : m_filePosition{filePos}, m_timeOffsetMs{timeOffsetMs}, m_deviceId{deviceId}
    {}

    virtual ~FrameIndexEntryIn6130() = default;

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    uint64_t getFilePosition() const { return m_filePosition; }
    uint64_t getTimeOffsetMs() const { return m_timeOffsetMs; }
    uint8_t getDeviceId() const { return m_deviceId; }

public:
    void setFilePosition(const uint64_t filePosition) { m_filePosition = filePosition; }
    void setTimeOffsetMs(const uint64_t timeOffsetMs) { m_timeOffsetMs = timeOffsetMs; }
    void setDeviceId(const uint8_t deviceId) { m_deviceId = deviceId; }

protected:
    uint64_t m_filePosition; ///< position of the frame in the file
    uint64_t m_timeOffsetMs; ///< time offset of this frame in milliseconds
    uint8_t m_deviceId; ///< device id that defined the frame
}; // FrameIndexEntryIn6130

//==============================================================================

bool operator==(const FrameIndexEntryIn6130& lhs, const FrameIndexEntryIn6130& rhs);
bool operator!=(const FrameIndexEntryIn6130& lhs, const FrameIndexEntryIn6130& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
