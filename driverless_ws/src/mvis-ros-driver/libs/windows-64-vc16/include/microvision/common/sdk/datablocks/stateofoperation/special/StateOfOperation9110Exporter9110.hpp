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
//! \date Mar 16, 2018
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/stateofoperation/special/StateOfOperation9110.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Serialize StateOfOperation9110 data container into stream.
//------------------------------------------------------------------------------
template<>
class Exporter<StateOfOperation9110, DataTypeId::DataType_StateOfOperation9110>
  : public TypedExporter<StateOfOperation9110, DataTypeId::DataType_StateOfOperation9110>
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
}; // StateOfOperation9110Exporter9110

//==============================================================================

using StateOfOperation9110Exporter9110 = Exporter<StateOfOperation9110, DataTypeId::DataType_StateOfOperation9110>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
