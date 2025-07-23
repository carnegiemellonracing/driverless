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
//! \date Feb 04, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/stateofoperation/StateOfOperation.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Serialize StateOfOperation data container into binary data stream.
//------------------------------------------------------------------------------
template<>
class Exporter<StateOfOperation, DataTypeId::DataType_StateOfOperation9111>
  : public TypedExporter<StateOfOperation, DataTypeId::DataType_StateOfOperation9111>
{
public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    Exporter() = default;

    //========================================
    //! \brief Copy construction not allowed!
    //----------------------------------------
    Exporter(const Exporter&) = delete;

    //========================================
    //! \brief No assignment construction allowed!
    //----------------------------------------
    Exporter& operator=(const Exporter&) = delete;

public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // StateOfOperationExporter9111

//==============================================================================

using StateOfOperationExporter9111 = Exporter<StateOfOperation, DataTypeId::DataType_StateOfOperation9111>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
