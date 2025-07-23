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
//! \date Feb 4, 2019
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainer.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This is the exporter for custom data containers.
//!
//! This can export any custom data container (even with unknown internals)
//! as a header/byte content data block.
//------------------------------------------------------------------------------
template<>
class Exporter<CustomDataContainer, DataTypeId::DataType_CustomDataContainer>
  : public TypedExporter<CustomDataContainer, DataTypeId::DataType_CustomDataContainer>
{
public:
    //========================================
    //! \brief Calculate the serialized size of the written data container
    //!
    //! \param[in] c  Custom data container to be serialized.
    //! \return The size in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    //========================================
    //! \brief Serialize the custom data container to stream
    //!
    //! \param[in, out] os  The IDC output Stream.
    //! \param[in] c    Custom data container to be serialized.
    //! \return True if written, false if something went wrong.
    //----------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // CustomDataContainerExporter

//==============================================================================

using CustomDataContainerExporter = Exporter<CustomDataContainer, DataTypeId::DataType_CustomDataContainer>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
