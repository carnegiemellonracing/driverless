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
//! \date Apr 27, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectList.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Importer<ObjectList, DataTypeId::DataType_ObjectList2271>
  : public RegisteredImporter<ObjectList, DataTypeId::DataType_ObjectList2271>
{
public: // implements ImporterBase
    //=================================================
    //! \brief Get the size in bytes that the object occupies when being serialized.
    //!
    //! \param[in] dataContainer  Object to get the serialized size for.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //! \return  the number of bytes used for serialization.
    //-------------------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& dataContainer,
                                      const ConfigurationPtr& configuration = nullptr) const override;

    //=================================================
    //! \brief Read data from the given stream and fill the given data container (deserialization).
    //!
    //! \param[in, out] inputStream     Input data stream
    //! \param[out]     dataContainer   Output container defining the target type (might include conversion).
    //! \param[in]      dataHeader      Metadata prepended to each idc data block.
    //! \param[in]      configuration   (Optional) Configuration context for import. Default set as nullptr.
    //! \return \c True if deserialization succeeds, \c false otherwise.
    //!
    //! \note This method has to be called from outside for deserialization.
    //-------------------------------------------------
    bool deserialize(std::istream& inputStream,
                     DataContainerBase& dataContainer,
                     const IdcDataHeader& dataHeader,
                     const ConfigurationPtr& configuration = nullptr) const override;

private:
    static const uint8_t attributesFlagUnfilteredAttributesContourAvailable{0x01};
    static const uint8_t attributesFlagUnfilteredAttributesAvailable{0x02};
    static const uint8_t attributesFlagFilteredAttributesContourAvailable{0x04};
    static const uint8_t attributesFlagFilteredAttributesAvailable{0x08};

private:
    static constexpr float yawRateScalingFactor{10000.0F};
    static constexpr float objectBoxHeightScalingFactor{3.0F};

    // Deserialize the general object in context of a data container type 0x2271.
    std::streamsize getSerializedSize(const Object& object) const;
    bool deserialize(std::istream& is, Object& object, const NtpTime& containerTimestamp) const;

    std::streamsize getSerializedSize(const UnfilteredObjectData& objectData) const;
    bool deserialize(std::istream& is,
                     UnfilteredObjectData& objectData,
                     const NtpTime& containerTimestamp,
                     const bool hasContourPoints) const;

    static std::streamsize getContourPointSerializedSize();
    bool deserialize(std::istream& is, ContourPoint& cpData) const;

    bool hasFilteredObjectData(const Object& object) const;
    void resetFilteredObjectData(Object& object) const;
}; // ObjectListImporter2271

//==============================================================================

using ObjectListImporter2271 = Importer<ObjectList, DataTypeId::DataType_ObjectList2271>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
