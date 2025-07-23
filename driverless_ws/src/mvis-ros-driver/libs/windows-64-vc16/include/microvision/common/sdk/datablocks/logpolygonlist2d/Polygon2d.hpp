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
//! \date May 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<typename T>
class Polygon2d final : public std::vector<Vector2<T>>
{
public:
    using ValueType = T;

public:
    Polygon2d() : std::vector<Vector2<ValueType>>() {}

    Polygon2d(const Vector2<ValueType>& p1) : std::vector<Vector2<ValueType>>(1, p1) {}

    Polygon2d(const Vector2<ValueType>& p1, const Vector2<ValueType>& p2) : std::vector<Vector2<ValueType>>()
    {
        this->reserve(2);
        std::vector<Vector2<T>>::push_back(p1);
        std::vector<Vector2<T>>::push_back(p2);
    }

    Polygon2d(const Vector2<ValueType>& p1, const Vector2<ValueType>& p2, const Vector2<ValueType>& p3)
      : std::vector<Vector2<ValueType>>()
    {
        this->reserve(3);
        std::vector<Vector2<T>>::push_back(p1);
        std::vector<Vector2<T>>::push_back(p2);
        std::vector<Vector2<T>>::push_back(p3);
    }

    Polygon2d(const Vector2<ValueType>& p1,
              const Vector2<ValueType>& p2,
              const Vector2<ValueType>& p3,
              const Vector2<ValueType>& p4)
      : std::vector<Vector2<ValueType>>()
    {
        this->reserve(4);
        push_back(p1);
        push_back(p2);
        push_back(p3);
        push_back(p4);
    }

    virtual ~Polygon2d() = default;

public:
    virtual std::streamsize getSerializedSize() const
    {
        return std::streamsize(sizeof(uint16_t))
               + std::streamsize(this->size()) * microvision::common::sdk::serializedSize(Vector2<ValueType>());
    }

    virtual bool deserialize(std::istream& is)
    {
        const int64_t startPos = streamposToInt64(is.tellg());

        uint16_t sz;
        readBE(is, sz);
        this->resize(sz);

        for (Vector2<ValueType>& element : *this)
        {
            readBE(is, element);
        }

        return !is.fail() && ((streamposToInt64(is.tellg()) - startPos) == getSerializedSize());
    }

    virtual bool serialize(std::ostream& os) const
    {
        const int64_t startPos = streamposToInt64(os.tellp());

        const uint16_t sz = uint16_t(this->size());
        writeBE(os, sz);

        for (const Vector2<ValueType>& element : *this)
        {
            writeBE(os, element);
        }

        return !os.fail() && ((streamposToInt64(os.tellp()) - startPos) == getSerializedSize());
    }
}; // Polygon2d

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
