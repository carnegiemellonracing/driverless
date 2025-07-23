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
//! \date Apr 26, 2012
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ResolutionInfoIn2205 final
{
public:
    ResolutionInfoIn2205();
    ResolutionInfoIn2205(const ResolutionInfoIn2205& src);
    virtual ~ResolutionInfoIn2205() = default;

public:
    ResolutionInfoIn2205& operator=(const ResolutionInfoIn2205& src);

public:
    static std::streamsize getSerializedSize_static();

public:
    virtual std::streamsize getSerializedSize() const { return getSerializedSize_static(); }

    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public: // getter
    float getResolutionStartAngle() const { return m_resolutionStartAngle; }
    float getResolution() const { return m_resolution; }

public: // setter
    void setResolutionStartAngle(const float newResolutionStartAngle)
    {
        m_resolutionStartAngle = newResolutionStartAngle;
    }
    void setResolution(const float newResolution) { m_resolution = newResolution; }

public:
    bool operator==(const ResolutionInfoIn2205& other) const;
    bool operator!=(const ResolutionInfoIn2205& other) const { return !((*this) == other); }

protected:
    //! Starting from this angle the given resolution is valid until the next resolution start angle or the scan end.
    //! In radians normalized to  [-\pi,+\pi[ . Valid only if resolution value is > 0.
    float m_resolutionStartAngle;

    //! Resolution for this sector.
    //! In radians normalized to  [-\pi,+\pi[ . Valid only if resolution value is > 0.
    float m_resolution;
}; // ResolutionInfo

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
