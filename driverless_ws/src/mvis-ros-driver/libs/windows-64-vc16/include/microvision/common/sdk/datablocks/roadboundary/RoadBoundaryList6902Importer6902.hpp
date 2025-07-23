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
//! \date Okt 19, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundaryList6902.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This is the importer for road boundaries.
//!
//! Deserializes a bytestream into a RoadBoundaryList6902.
//------------------------------------------------------------------------------
template<>
class Importer<RoadBoundaryList6902, static_cast<DataTypeId::DataType>(DataTypeId::DataType_RoadBoundaryList6902)>
  : public RegisteredImporter<RoadBoundaryList6902,
                              static_cast<DataTypeId::DataType>(DataTypeId::DataType_RoadBoundaryList6902)>
{
public:
    //========================================
    //! \brief Empty constructor calling base.
    //----------------------------------------
    Importer() : RegisteredImporter() {}

    //========================================
    //! \brief Copy construction is forbidden.
    //----------------------------------------
    Importer(const Importer&) = delete;

    //========================================
    //! \brief Assignment construction is forbidden.
    //----------------------------------------
    Importer& operator=(const Importer&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Importer() override = default;

public:
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
    //========================================
    //!\brief Convert from byte stream (deserialization).
    //!\param[in]      is  Input data stream
    //!\param[in, out] boundary   Data container.
    //!\return \c True if deserialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool deserialize(std::istream& is, RoadBoundaryIn6902& boundary);

    //========================================
    //!\brief Convert from byte stream (deserialization).
    //!\param[in]      is  Input data stream
    //!\param[in, out] c   Data container.
    //!\return \c True if deserialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool deserialize(std::istream& is, RoadBoundarySegmentIn6902& c);

    //========================================
    //!\brief Convert from byte stream (deserialization).
    //!\param[in]      is  Input data stream
    //!\param[in, out] c   Data container.
    //!\return \c True if deserialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool deserialize(std::istream& is, RoadBoundaryPointIn6902& c);
}; //RoadBoundaryList6902Importer6902

//==============================================================================

using RoadBoundaryList6902Importer6902
    = Importer<RoadBoundaryList6902, static_cast<DataTypeId::DataType>(DataTypeId::DataType_RoadBoundaryList6902)>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
