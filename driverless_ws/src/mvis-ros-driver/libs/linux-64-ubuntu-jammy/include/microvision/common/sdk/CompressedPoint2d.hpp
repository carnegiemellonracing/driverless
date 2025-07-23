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
//! \date Sep 13, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/Vector2.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CompressedPoint2d final : private Vector2<int16_t>
{
public:
    static const int16_t compressedPlusInfinity;
    static const int16_t compressedMinusInfinity;
    static const int16_t compressedNotANumber;

    static const float toCentimeter;
    static const float fromCentimeter;
    static const float toDecimeter;
    static const float fromDecimeter;

public:
    static std::streamsize getSerializedSize_static() { return serializedSize(Vector2<int16_t>{}); }
    static float decompressMeters(const int16_t compressedMeters);
    static float decompressRadians(const int16_t compressedRadians);

    static int16_t compressMeters(const float meters);
    static int16_t compressRadians(const float radians);

private:
    static constexpr uint8_t indexOfX{0U};
    static constexpr uint8_t indexOfY{1U};

public:
    CompressedPoint2d();
    CompressedPoint2d(const int16_t compressedPosX, const int16_t compressedPosY);

    ~CompressedPoint2d();

public:
    CompressedPoint2d& operator=(const CompressedPoint2d& src);

public:
    int16_t getCompressedX() const { return getX(); }
    int16_t getCompressedY() const { return getY(); }

    float getXinMeter() const { return decompressMeters(getX()); }
    float getYinMeter() const { return decompressMeters(getY()); }

public:
    void setCompressedX(const int16_t compressedPosX) { setX(compressedPosX); }
    void setCompressedY(const int16_t compressedPosY) { setY(compressedPosY); }

    void setX(const float posXMeter) { Vector2<int16_t>::setX(compressMeters(posXMeter)); }
    void setY(const float posYMeter) { Vector2<int16_t>::setY(compressMeters(posYMeter)); }

public:
    template<typename T>
    friend void readLE(std::istream& is, T& value);
    template<typename T>
    friend void readBE(std::istream& is, T& value);
    template<typename T>
    friend void writeLE(std::ostream& os, const T& value);
    template<typename T>
    friend void writeBE(std::ostream& os, const T& value);

}; // CompressedPoint2d

//==============================================================================
// Specializations for operator
//==============================================================================

//==============================================================================
//! \brief Stream operator for writing the point content to a stream.
//! \param[in] os  The stream, the vector shall be written to.
//! \param[in] pt  The point which shall be streamed.
//! \return The stream to which the vector was written to.
//------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const Vector2<int16_t>& pt);

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const CompressedPoint2d& lhs, const CompressedPoint2d& rhs);

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator!=(const CompressedPoint2d& lhs, const CompressedPoint2d& rhs);

//==============================================================================
// Specializations for Serialization
//==============================================================================

template<>
inline void writeLE<CompressedPoint2d>(std::ostream& os, const CompressedPoint2d& value)
{
    microvision::common::sdk::writeLE(os, value.m_elements[CompressedPoint2d::indexOfX]);
    microvision::common::sdk::writeLE(os, value.m_elements[CompressedPoint2d::indexOfY]);
}

//==============================================================================

template<>
inline void readLE<CompressedPoint2d>(std::istream& is, CompressedPoint2d& value)
{
    microvision::common::sdk::readLE(is, value.m_elements[CompressedPoint2d::indexOfX]);
    microvision::common::sdk::readLE(is, value.m_elements[CompressedPoint2d::indexOfY]);
}

//==============================================================================

template<>
inline void writeBE<CompressedPoint2d>(std::ostream& os, const CompressedPoint2d& value)
{
    microvision::common::sdk::writeBE(os, value.m_elements[CompressedPoint2d::indexOfX]);
    microvision::common::sdk::writeBE(os, value.m_elements[CompressedPoint2d::indexOfY]);
}

//==============================================================================

template<>
inline void readBE<CompressedPoint2d>(std::istream& is, CompressedPoint2d& value)
{
    microvision::common::sdk::readBE(is, value.m_elements[CompressedPoint2d::indexOfX]);
    microvision::common::sdk::readBE(is, value.m_elements[CompressedPoint2d::indexOfY]);
}

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
