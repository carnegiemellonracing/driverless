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

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/stateofoperation/special/StateOfOperation9110.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Deserialize a stream into a StateOfOperation9110 data container.
//------------------------------------------------------------------------------
template<>
class Importer<StateOfOperation9110, DataTypeId::DataType_StateOfOperation9110>
  : public RegisteredImporter<StateOfOperation9110, DataTypeId::DataType_StateOfOperation9110>
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
    Importer() : RegisteredImporter() {}

    //========================================
    //! \brief Copy construction not allowed!
    //----------------------------------------
    Importer(const Importer&) = delete;

    //========================================
    //! \brief No assignment construction allowed!
    //----------------------------------------
    Importer& operator=(const Importer&) = delete;

public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c, const ConfigurationPtr&) const override;

    //========================================
    //!\brief convert data from source to target type (deserialization)
    //!\param[in, out] is      Input data stream
    //!\param[out]     c       Output container.
    //!\param[in]      header  idc dataHeader
    //!\return \c True if serialization succeed, else: \c false
    //!\note This method is to be called from outside for deserialization.
    //----------------------------------------
    bool
    deserialize(std::istream& is, DataContainerBase& c, const IdcDataHeader&, const ConfigurationPtr&) const override;
}; //StateOfOperation9110Importer9110

//==============================================================================

using StateOfOperation9110Importer9110
    = Importer<microvision::common::sdk::StateOfOperation9110, DataTypeId::DataType_StateOfOperation9110>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
