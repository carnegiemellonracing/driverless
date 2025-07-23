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
//! \date Nov 22, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This helper class is used to get the serialized size of all sub data containers used by the
//!        (de)serialization of Scan2340 data container from and into a scan2340 stream.
//------------------------------------------------------------------------------
class Scan2340SerializedSize2340
{
public:
    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of Scan2340 data container from and into a
    //!         scan2340 stream.
    //----------------------------------------
    static std::streamsize getSerializedSize(const Scan2340& scan);

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScannerInfoIn2340 data container from and into a
    //!         scan2340 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSize(const ScannerInfoIn2340&);

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScanPointIn2340 data container from and into a
    //!         scan2340 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSize(const ScanPointIn2340&);

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScanPointInfoListIn2340 data container from and into a
    //!         scan2340 stream.
    //----------------------------------------
    static std::streamsize getSerializedSize(const ScanPointInfoListIn2340& pointInfoList);

}; //Scan2340SerializedSize2340

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
