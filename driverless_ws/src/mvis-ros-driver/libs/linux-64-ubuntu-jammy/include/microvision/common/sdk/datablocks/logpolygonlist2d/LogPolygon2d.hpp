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

#include <microvision/common/sdk/datablocks/logpolygonlist2d/Polygon2d.hpp>
#include <microvision/common/sdk/datablocks/LogPolygonMessage.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<typename T>
class LogPolygon2d final
{
public:
    using ValueType = T;

public:
    LogPolygon2d()          = default;
    virtual ~LogPolygon2d() = default;

public:
    virtual std::streamsize getSerializedSize() const
    {
        return std::streamsize(sizeof(uint32_t) + m_msg.getRawMessage().size() + 1) + m_poly.getSerializedSize();
    }

    virtual bool deserialize(std::istream& is)
    {
        const int64_t startPos = streamposToInt64(is.tellg());

        uint32_t sizeInBytes = 0;
        readBE(is, sizeInBytes);

        if (!m_poly.deserialize(is))
            return false;

        // the rest of the package is the string
        const std::streamsize stringSize = std::streamsize(sizeInBytes) - m_poly.getSerializedSize();
        char* buf                        = new char[std::size_t(stringSize)];
        is.read(buf, stringSize);
        std::string msg;
        msg.assign(buf, uint32_t(stringSize - 1));
        m_msg.setRawMessage(std::move(msg));
        delete[] buf;

        return !is.fail() && ((streamposToInt64(is.tellg()) - startPos) == this->getSerializedSize());
    }

    virtual bool serialize(std::ostream& os) const
    {
        const int64_t startPos = streamposToInt64(os.tellp());

        const uint32_t serSz = uint32_t(m_msg.getRawMessage().size() + 1 + uint32_t(m_poly.getSerializedSize()));
        writeBE(os, serSz);

        m_poly.serialize(os);

        os << m_msg.getRawMessage();

        // Add termination char
        writeBE(os, uint8_t(std::string::value_type(0)));

        return !os.fail() && ((streamposToInt64(os.tellp()) - startPos) == this->getSerializedSize());
    }

public:
    const LogPolygonMessage& getMsg() const { return m_msg; }
    const Polygon2d<ValueType>& getPoly() const { return m_poly; }

public:
    void setMsg(const LogPolygonMessage& msg) { m_msg = msg; }
    void setPoly(const Polygon2d<ValueType>& poly) { m_poly = poly; }

private:
    Polygon2d<ValueType> m_poly;
    LogPolygonMessage m_msg;
}; // LogPolygon2d

//==============================================================================

template<typename T>
inline bool operator==(const LogPolygon2d<T>& lhs, const LogPolygon2d<T>& rhs)
{
    return (lhs.getPoly() == rhs.getPoly()) && (lhs.getMsg().getRawMessage() == rhs.getMsg().getRawMessage());
}

template<typename T>
inline bool operator!=(const LogPolygon2d<T>& lhs, const LogPolygon2d<T>& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
