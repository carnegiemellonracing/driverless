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
//! \date Jul 31, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayList6972.hpp>
#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Importer<CarriageWayList6972, DataTypeId::DataType_CarriageWayList6972>
  : public RegisteredImporter<CarriageWayList6972, DataTypeId::DataType_CarriageWayList6972>
{
public:
    Importer()                = default;
    Importer(const Importer&) = delete;
    Importer& operator=(const Importer&) = delete;

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

public:
    static bool deserialize(std::istream& is, lanes::CarriageWayIn6972& cw);
    static bool deserialize(std::istream& is, lanes::CarriageWaySegmentIn6972& cws);
    static bool deserialize(std::istream& is, lanes::LaneIn6972& lane);
    static bool deserialize(std::istream& is, lanes::LaneSegmentIn6972& laneSeg);
    static bool deserialize(std::istream& is, lanes::LaneSupportPointIn6972& point);
}; //CarriageWayList6972Importer6972

//==============================================================================

using CarriageWayList6972Importer6972 = Importer<CarriageWayList6972, DataTypeId::DataType_CarriageWayList6972>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
