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

#include <vector>

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/DataPackage.hpp>
#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Raw stream data package.
//!
//! Package of a specific SDK datatype in serialized form.
//! This class is the specialization of data package which added the \sa IdcDataHeader
//! to identifying the SDK datatype.
//!
//! Example: Two idc-files from two different trips which contains Lidar scan
//! data shall be processed by an algorithm. This algorithm creates a new data package stream
//! as output. This stream is ordered by index and the source URI is takeover by original.
//------------------------------------------------------------------------------
class IdcDataPackage : public DataPackage
{
public:
    //========================================
    //! \brief Constructor with index and the source Uri.
    //! \param[in] index  Position in the data package stream, depends to source.
    //! \param[in] uri    Source of data (stream).
    //----------------------------------------
    IdcDataPackage(const int64_t index, const Uri& uri) : DataPackage{index, uri}, m_header{} {}

    //========================================
    //! \brief Constructor with index, source Uri and IDC header.
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source Uri of data (stream).
    //! \param[in] header   IdcDataHeader of package.
    //----------------------------------------
    IdcDataPackage(const int64_t index, const Uri& uri, const IdcDataHeader& header)
      : DataPackage{index, uri}, m_header{header}
    {}

    //========================================
    //! \brief Constructor with index, the source Uri, IDC header and binary data.
    //! \param[in] index    Position in the data package stream, depends to source.
    //! \param[in] uri      Source Uri of data (stream).
    //! \param[in] header   IdcDataHeader of package.
    //! \param[in] payload  Binary data chunk.
    //----------------------------------------
    IdcDataPackage(const int64_t index, const Uri& uri, const IdcDataHeader& header, const PayloadType& payload)
      : DataPackage{index, uri, payload}, m_header{header}
    {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcDataPackage() override = default;

public:
    //========================================
    //! \brief Compare two idc data packages for equality.
    //! \param[in] lhs  IdcDataPackage to compare.
    //! \param[in] rhs  IdcDataPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const IdcDataPackage& lhs, const IdcDataPackage& rhs)
    {
        return (lhs.getPayload() == rhs.getPayload())
               && (lhs.getHeader().getDataType() == rhs.getHeader().getDataType())
               && (lhs.getHeader().getDeviceId() == rhs.getHeader().getDeviceId())
               && (lhs.getHeader().getMessageSize() == rhs.getHeader().getMessageSize())
               && (lhs.getHeader().getTimestamp() == rhs.getHeader().getTimestamp());
    }

    //========================================
    //! \brief Compare two idc data packages for inequality.
    //! \param[in] lhs  IdcDataPackage to compare.
    //! \param[in] rhs  IdcDataPackage to compare.
    //! \note Offset wont compare because of section compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const IdcDataPackage& lhs, const IdcDataPackage& rhs) { return !(lhs == rhs); }

public: //getter
    //========================================
    //! \brief Get IdcDataHeader of package.
    //! \return IdcDataHeader of package.
    //----------------------------------------
    IdcDataHeader& getHeader() { return this->m_header; }

    //========================================
    //! \brief Get IdcDataHeader of package.
    //! \return IdcDataHeader of package.
    //----------------------------------------
    const IdcDataHeader& getHeader() const { return this->m_header; }

public: //setter
    //========================================
    //! \brief Set IdcDataHeader of package.
    //! \param[in] dataHeader  IdcDataHeader of package.
    //----------------------------------------
    void setHeader(const IdcDataHeader& dataHeader) { this->m_header = dataHeader; }

private:
    //========================================
    //! \brief IdcDataHeader of package.
    //----------------------------------------
    IdcDataHeader m_header;
};

//==============================================================================

//========================================
//! \brief Nullable IdcDataPackage pointer.
//----------------------------------------
using IdcDataPackagePtr = std::shared_ptr<IdcDataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
