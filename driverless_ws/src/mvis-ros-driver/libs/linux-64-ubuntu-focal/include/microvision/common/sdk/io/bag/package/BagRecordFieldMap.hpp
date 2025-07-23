//==============================================================================
//! \file
//!
//! \brief BAG record field map for header definitions.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 10, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>

#include <microvision/common/sdk/misc/SharedBufferStream.hpp>
#include <microvision/common/sdk/misc/SharedBuffer.hpp>
#include <microvision/common/sdk/Ptp64Time.hpp>
#include <microvision/common/sdk/io.hpp>

#include <microvision/common/logging/logging.hpp>

#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief BAG record field map for header definitions.
//------------------------------------------------------------------------------
class BagRecordFieldMap final
{
public: // type definitions
    //========================================
    //! \brief Map to store record header.
    //----------------------------------------
    using FieldMap = std::map<std::string, SharedBuffer>;

    //========================================
    //! \brief BAG file record header field delimiter character.
    //----------------------------------------
    static constexpr char fieldDelimiter{'='};

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagRecordFieldMap";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    BagRecordFieldMap();

    //========================================
    //! \brief Construct BAG record field map with key value pairs.
    //! \param[in] fields  Key value pairs.
    //----------------------------------------
    BagRecordFieldMap(std::initializer_list<typename FieldMap::value_type> fields);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagRecordFieldMap();

public:
    //========================================
    //! \brief Compare two bag record field maps for equality.
    //! \param[in] lhs  BagRecordFieldMap to compare.
    //! \param[in] rhs  BagRecordFieldMap to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const BagRecordFieldMap& lhs, const BagRecordFieldMap& rhs);

    //========================================
    //! \brief Compare two bag record field maps for inequality.
    //! \param[in] lhs  BagRecordPackage to compare.
    //! \param[in] rhs  BagRecordPackage to compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const BagRecordFieldMap& lhs, const BagRecordFieldMap& rhs);

public:
    //========================================
    //! \brief Get number of entries in field map.
    //! \returns Number of fields.
    //----------------------------------------
    std::size_t size() const;

    //========================================
    //! \brief Find map entry by field name.
    //! \param[in] fieldName  Name of the field entry in map.
    //! \returns Either map \c iterator to field of name if exists, otherwise \c end().
    //----------------------------------------
    FieldMap::iterator find(const std::string& fieldName);

    //========================================
    //! \brief Find map entry by field name.
    //! \param[in] fieldName  Name of the field entry in map.
    //! \returns Either map \c iterator to field of name if exists, otherwise \c end().
    //----------------------------------------
    FieldMap::const_iterator find(const std::string& fieldName) const;

    //========================================
    //! \brief Get begin entry of field map.
    //! \returns Either map \c iterator to first entry if not empty, otherwise \c end().
    //----------------------------------------
    FieldMap::iterator begin();

    //========================================
    //! \brief Get begin entry of field map.
    //! \returns Either map \c iterator to first entry if not empty, otherwise \c end().
    //----------------------------------------
    FieldMap::const_iterator begin() const;

    //========================================
    //! \brief Get end pointer after last entry of field map.
    //! \returns Pointer after last entry.
    //----------------------------------------
    FieldMap::iterator end();

    //========================================
    //! \brief Get end pointer after last entry of field map.
    //! \returns Pointer after last entry.
    //----------------------------------------
    FieldMap::const_iterator end() const;

    //========================================
    //! \brief Remove all entries from field map.
    //----------------------------------------
    void clear();

public: // getter
    //========================================
    //! \brief Get value from header key value map.
    //! \param[in] fieldName  Key of header map.
    //! \returns Value of header map.
    //----------------------------------------
    template<typename ValueType>
    ValueType getFieldValueAs(const std::string& fieldName, const ValueType defaultValue = ValueType{}) const;

    //========================================
    //! \brief Get value from header key value map.
    //! \param[in] fieldName  Key of header map.
    //! \returns Value of header map.
    //----------------------------------------
    std::string getFieldValueAsString(const std::string& fieldName) const;

    //========================================
    //! \brief Get value from header key value map.
    //! \param[in] fieldName  Key of header map.
    //! \returns Value of header map.
    //----------------------------------------
    Ptp64Time getFieldValueAsPtp64Time(const std::string& fieldName) const;

public: // setter
    //========================================
    //! \brief Set value on header key value map.
    //! \param[in] fieldName    Key of header map.
    //! \param[in] data         New value of header map.
    //! \param[in] dataLength   Length of new value length.
    //----------------------------------------
    template<typename ValueType>
    void setFieldValueAs(const std::string& fieldName, const ValueType fieldValue);

    //========================================
    //! \brief Get value from header key value map.
    //! \param[in] fieldName  Key of header map.
    //! \returns Value of header map.
    //----------------------------------------
    void setFieldValueAsString(const std::string& fieldName, const std::string& fieldValue);

    //========================================
    //! \brief Get value from header key value map.
    //! \param[in] fieldName  Key of header map.
    //! \returns Value of header map.
    //----------------------------------------
    void setFieldValueAsPtp64Time(const std::string& fieldName, const Ptp64Time& fieldValue);

public:
    //========================================
    //! \brief Compute serialized size from record field map.
    //! \return Length of serialized field map or \c 0 if is empty.
    //----------------------------------------
    std::size_t computeSize() const;

public:
    //========================================
    //! \brief Read a value of type \a BagRecordFieldMap from a stream.
    //!        Reading individual bytes is done in little endian byte order.
    //! \param[in, out] is     Stream providing the data to be read.
    //! \note Called by inline template function \c readLE to grant access to private members.
    //!       To avoid an error in clang a friend declaration is not possible.
    //----------------------------------------
    void deserializeLE(std::istream& is);

    //========================================
    //! \brief Write a value of type \a BagRecordFieldMap into a stream.
    //!        Writing individual bytes is done in little endian byte order.
    //! \param[in, out] os     Stream that will receive the data to be written.
    //! \note Called by inline template function \c readLE to grant access to private members.
    //!       To avoid an error in clang a friend declaration is not possible.
    //----------------------------------------
    void serializeLE(std::ostream& os) const;

private:
    //========================================
    //! \brief BAG record field map.
    //----------------------------------------
    FieldMap m_fields;

}; // class BagRecordFieldMap

//==============================================================================
//! \brief Read a value of type \a BagRecordFieldMap from a stream.
//!        Reading individual bytes is done in little endian byte order.
//! \param[in, out] is     Stream providing the data to be read.
//! \param[out]     value  On exit it will hold the value that has been read.
//------------------------------------------------------------------------------
template<>
inline void readLE<BagRecordFieldMap>(std::istream& is, BagRecordFieldMap& value)
{
    value.deserializeLE(is);
}

//==============================================================================
//! \brief Write a value of type \a BagRecordFieldMap into a stream.
//!        Writing individual bytes is done in little endian byte order.
//! \param[in, out] os     Stream that will receive the data to be written.
//! \param[in]      value  The value to be written.
//------------------------------------------------------------------------------
template<>
inline void writeLE<BagRecordFieldMap>(std::ostream& os, const BagRecordFieldMap& value)
{
    value.serializeLE(os);
}

//==============================================================================

template<typename ValueType>
ValueType BagRecordFieldMap::getFieldValueAs(const std::string& fieldName, const ValueType defaultValue) const
{
    const auto fieldIterator = this->m_fields.find(fieldName);

    if (fieldIterator != this->m_fields.end())
    {
        SharedBufferStream stream{fieldIterator->second};
        ValueType value;

        readLE(stream, value);

        if (stream.operator bool())
        {
            return value;
        }
    }

    return defaultValue;
}

//==============================================================================

template<typename ValueType>
void BagRecordFieldMap::setFieldValueAs(const std::string& fieldName, const ValueType fieldValue)
{
    auto fieldIterator = this->m_fields.find(fieldName);

    if (fieldIterator == this->m_fields.end())
    {
        // field does not yet exist -> add a new one
        fieldIterator = this->m_fields.insert(std::make_pair(fieldName, SharedBuffer{})).first;
    }

    if (fieldIterator->second.size() != sizeof(ValueType))
    {
        // resize buffer to fit new value type
        fieldIterator->second.resize(sizeof(ValueType));
    }

    SharedBufferStream stream{fieldIterator->second};

    writeLE(stream, fieldValue);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
