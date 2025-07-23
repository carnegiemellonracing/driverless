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
#include <microvision/common/sdk/datablocks/stateofoperation/special/StateOfOperation9111.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Serialize StateOfOperation9111 data container into stream.
//------------------------------------------------------------------------------
template<>
class Exporter<StateOfOperation9111, DataTypeId::DataType_StateOfOperation9111>
  : public TypedExporter<StateOfOperation9111, DataTypeId::DataType_StateOfOperation9111>
{
public:
    //========================================
    //! \brief Serialized size of this data container.
    //----------------------------------------
    static constexpr std::streamsize serializedBaseSize{19};

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
}; // StateOfOperation9111Exporter9111

//==============================================================================

using StateOfOperation9111Exporter9111 = Exporter<StateOfOperation9111, DataTypeId::DataType_StateOfOperation9111>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
