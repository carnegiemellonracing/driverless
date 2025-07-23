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
//! \date Mar 14, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/measurementlist/special/MeasurementKeyIn2821.hpp>

#include <microvision/common/logging/logging.hpp>

#include <iostream>
#include <boost/any.hpp>
#include <boost/assert.hpp>

#include <stdexcept>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class MeasurementIn2821 final
{
public:
    enum class MeasurementType : uint8_t
    {
        Void      = 0x00U,
        Float     = 0x01U,
        Double    = 0x02U,
        INT8      = 0x03U,
        UINT8     = 0x04U,
        INT16     = 0x05U,
        UINT16    = 0x06U,
        INT32     = 0x07U,
        UINT32    = 0x08U,
        INT64     = 0x09U,
        UINT64    = 0x0AU,
        Bool      = 0x0BU,
        StdString = 0x0CU
    };

public:
    MeasurementIn2821();
    virtual ~MeasurementIn2821() = default;

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public: // getter
    MeasurementKeyIn2821 getKey() const { return m_key; }
    MeasurementType getMeasurementType() const { return typeIdToType(getType()); }

    template<typename T>
    T getData() const
    {
        return boost::any_cast<T>(m_data);
    }

public: // setter
    void setKey(const MeasurementKeyIn2821 key) { m_key = key; }
    template<typename T>
    void setData(const T& data)
    {
        m_data = data;
    }

    template<typename T>
    void setValue(const MeasurementKeyIn2821 key, const T& value)
    {
        m_key  = key;
        m_data = value;
    }

    void resetValue() { m_data = boost::any(); }

    //========================================

public:
    template<typename T>
    T getAs() const
    {
        throw std::runtime_error("Only specializations are allowed.");
    }

    bool isEqual(const MeasurementIn2821& m) const;

    template<typename T>
    bool isOfType() const
    {
        return m_data.type() == typeid(T);
    }

    static MeasurementType typeIdToType(const std::type_info& tinfo);
    const std::type_info& getType() const { return m_data.type(); }

protected:
    static constexpr const char* loggerId = "microvision::common::sdk::Measurement";
    static microvision::common::logging::LoggerSPtr logger;

protected:
    MeasurementKeyIn2821 m_key;
    //MeasurementType m_measurementType;
    boost::any m_data;
}; // Measurement

//==============================================================================
template<>
void MeasurementIn2821::getAs<void>() const;

template<>
float MeasurementIn2821::getAs<float>() const;

template<>
double MeasurementIn2821::getAs<double>() const;

template<>
int8_t MeasurementIn2821::getAs<int8_t>() const;

template<>
uint8_t MeasurementIn2821::getAs<uint8_t>() const;

template<>
int16_t MeasurementIn2821::getAs<int16_t>() const;

template<>
uint16_t MeasurementIn2821::getAs<uint16_t>() const;

template<>
int32_t MeasurementIn2821::getAs<int32_t>() const;

template<>
uint32_t MeasurementIn2821::getAs<uint32_t>() const;

template<>
int64_t MeasurementIn2821::getAs<int64_t>() const;

template<>
uint64_t MeasurementIn2821::getAs<uint64_t>() const;

template<>
bool MeasurementIn2821::getAs<bool>() const;

template<>
std::string MeasurementIn2821::getAs<std::string>() const;
//==============================================================================

bool operator==(const MeasurementIn2821& lhs, const MeasurementIn2821& rhs);
bool operator!=(const MeasurementIn2821& lhs, const MeasurementIn2821& rhs);

//==============================================================================

std::ostream& operator<<(std::ostream& oss, const MeasurementIn2821& m);

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
