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
//! \date Jun 5, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SharedBuffer.hpp>
#include <microvision/common/sdk/misc/Uri.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Binary chunk of a data stream.
//!
//! One of the main purposes of the SDK is to transfer data between different
//! entities (sensors, ECUs and files). The data can be chunked into discrete
//! packages, which are then exchanged separately.
//!
//! The binary chunk of data stream provided with the header information
//! of position in stream (index) and source address as Uri.
//------------------------------------------------------------------------------
class DataPackage
{
public:
    using PayloadType = SharedBuffer;

public:
    //========================================
    //! \brief Constructor with index and the source Uri.
    //! \param[in] index  Position in the data package stream, depends to source.
    //! \param[in] uri    Source of data (stream) as Uri.
    //----------------------------------------
    DataPackage(const int64_t index, const Uri& uri) : m_index{index}, m_source{uri}, m_payload{} {}

    //========================================
    //! \brief Constructor with index, source Uri and binary data (payload).
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source of data (stream).
    //! \param[in] payload  Binary data chunk.
    //----------------------------------------
    DataPackage(const int64_t index, const Uri& uri, const PayloadType& payload)
      : m_index{index}, m_source{uri}, m_payload{payload}
    {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~DataPackage() = default;

public:
    //========================================
    //! \brief Compare data packages for equality.
    //! \param[in] lhs  DataPackage to compare.
    //! \param[in] rhs  DataPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //! \note This compares only the payload.
    //----------------------------------------
    friend bool operator==(const DataPackage& lhs, const DataPackage& rhs)
    {
        return (lhs.getPayload() == rhs.getPayload());
    }

    //========================================
    //! \brief Compare data packages for inequality.
    //! \param[in] lhs  DataPackage to compare.
    //! \param[in] rhs  DataPackage to compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //! \note This compares only the payload.
    //----------------------------------------
    friend bool operator!=(const DataPackage& lhs, const DataPackage& rhs)
    {
        return (lhs.getPayload() != rhs.getPayload());
    }

public: //getter
    //========================================
    //! \brief Get index of package stream.
    //!
    //! The index represents the order over packages like an index in a package stream.
    //! In case of a idc file it is the file cursor, otherwise it could be an iterator or
    //! something else depending to the source.
    //!
    //! \return Index in package stream.
    //----------------------------------------
    int64_t getIndex() const { return m_index; }

    //========================================
    //! \brief Get Uri of package source.
    //! \return Package source Uri.
    //----------------------------------------
    Uri& getSource() { return m_source; }

    //========================================
    //! \brief Get Uri of package source.
    //! \return Package source Uri.
    //----------------------------------------
    const Uri& getSource() const { return m_source; }

    //========================================
    //! \brief Get package payload.
    //! \return Binary data chunk.
    //----------------------------------------
    PayloadType& getPayload() { return m_payload; }

    //========================================
    //! \brief Get package payload.
    //! \return Binary data chunk.
    //----------------------------------------
    const PayloadType& getPayload() const { return m_payload; }

public: //setter
    //========================================
    //! \brief Set index of package stream.
    //!
    //! The index represents the order over packages like an index in a package stream.
    //! In case of a idc file it is the file cursor, otherwise it could be an iterator or
    //! something else depending to the source.
    //!
    //! \param[in] idx  Index in package stream.
    //----------------------------------------
    void setIndex(const int64_t idx) { m_index = idx; }

    //========================================
    //! \brief Set package source Uri.
    //! \param[in] source  Package source Uri.
    //----------------------------------------
    void setSource(const Uri& source) { m_source = source; }

    //========================================
    //! \brief Set package payload.
    //! \param[in] payload  Binary data chunk.
    //----------------------------------------
    void setPayload(const PayloadType& payload) { m_payload = payload; }

private:
    //========================================
    //! \brief Index of the data package in stream.
    //----------------------------------------
    int64_t m_index;

    //========================================
    //! \brief Package source uri.
    //----------------------------------------
    Uri m_source;

    //========================================
    //! \brief Binary payload of package.
    //----------------------------------------
    PayloadType m_payload;
};

//==============================================================================

//========================================
//! \brief Nullable DataPackage pointer.
//----------------------------------------
using DataPackagePtr = std::shared_ptr<DataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
