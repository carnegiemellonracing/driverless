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
//! \date May 11, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectList.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<ObjectList, DataTypeId::DataType_ObjectList2271>
  : public TypedExporter<ObjectList, DataTypeId::DataType_ObjectList2271>
{
public:
    virtual ~Exporter() = default;

public:
    //========================================
    //!\brief get size in bytes of serialized data
    //!\param[in]      c       Data container.
    //!\return \c True if serialization succeed, else: \c false
    //!\note This method is to be called from outside for serialization.
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const;

    //========================================
    //!\brief convert to byte stream (serialization)
    //!\param[in, out] os      Output data stream
    //!\param[in]      c       Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    //----------------------------------------
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const;

private:
    static const uint8_t attributesFlagUnfilteredAttributesContourAvailable{0x01};
    static const uint8_t attributesFlagUnfilteredAttributesAvailable{0x02};
    static const uint8_t attributesFlagFilteredAttributesContourAvailable{0x04};
    static const uint8_t attributesFlagFilteredAttributesAvailable{0x08};

private:
    // Deserialize the general object in context of a data container type 0x2271.
    std::streamsize getSerializedSize(const Object& object) const;
    bool serialize(std::ostream& os, const Object& object, const NtpTime& containerTimestamp) const;

    std::streamsize getSerializedSize(const UnfilteredObjectData& objectData) const;
    bool serialize(std::ostream& os, const UnfilteredObjectData& objectData, const NtpTime& containerTimestamp) const;

    static std::streamsize getContourPointSerializedSize();
    bool serialize(std::ostream& os, const ContourPoint& cpData) const;

    bool hasFilteredObjectData(const Object& object) const;
}; // ObjectListExporter2271

//==============================================================================

using ObjectListExporter2271 = Exporter<ObjectList, DataTypeId::DataType_ObjectList2271>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
