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
//! \date Jan 18, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class is used to export a processed low bandwidth MOVIA scan from a scan data container to serialize it into a
//! binary idc data block.
//------------------------------------------------------------------------------
template<>
class Exporter<Scan2342, DataTypeId::DataType_Scan2342> : public TypedExporter<Scan2342, DataTypeId::DataType_Scan2342>
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
    static bool serialize(std::ostream& os, const ScannerDirectionListIn2342& c);

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const ScannerInfoIn2342& c);

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const ScanPointIn2342& c);

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //----------------------------------------
    static bool serialize(std::ostream& os, const ScanPointRowIn2342& c);

}; // Scan2342Exporter2342

//==============================================================================

using Scan2342Exporter2342 = Exporter<Scan2342, DataTypeId::DataType_Scan2342>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
