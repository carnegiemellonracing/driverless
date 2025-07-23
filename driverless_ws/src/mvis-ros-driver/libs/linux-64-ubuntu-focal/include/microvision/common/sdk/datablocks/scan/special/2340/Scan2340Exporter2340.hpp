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
//! \date Nov 21, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class is used to export a processed MOVIA scan from a scan data container to serialize it into a
//! binary idc data block.
//------------------------------------------------------------------------------
template<>
class Exporter<Scan2340, DataTypeId::DataType_Scan2340> : public TypedExporter<Scan2340, DataTypeId::DataType_Scan2340>
{
public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //!\note This method is to be called from outside for serialization.
    //----------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

private:
    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const ScannerInfoIn2340& c);

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const ScanPointIn2340& c);

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const ScanPointInfoListIn2340& c);

}; // Scan2340Exporter2340

//==============================================================================

using Scan2340Exporter2340 = Exporter<Scan2340, DataTypeId::DataType_Scan2340>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
