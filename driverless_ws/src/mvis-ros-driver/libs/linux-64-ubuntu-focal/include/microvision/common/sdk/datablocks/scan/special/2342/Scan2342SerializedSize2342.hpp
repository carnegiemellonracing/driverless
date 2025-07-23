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

#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This helper class is used to get the serialized size of all sub data containers used by the
//!        (de)serialization of Scan2342 data container from and into a scan2342 stream.
//------------------------------------------------------------------------------
class Scan2342SerializedSize2342
{
public:
    //========================================
    //! \brief Get the size of the serialization.
    //! \param[in] withScannerInfo  Either \c true to get scan size with scanner info,
    ///!                            otherwise \c false to get without.
    //! \return Number of bytes used by the (de)serialization of Scan2342 data container from and into a
    //!         Scan2342 stream.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScan(const bool withScannerInfo);

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScannerDirectionListIn2342 data container
    //!          from and into a Scan2342 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScannerDirectionList();

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScannerInfoIn2342 data container from and into a
    //!         Scan2342 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScannerInfo();

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScanPointRowIn2342 data container from and into a
    //!         scan2342 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScanPointRow();

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScanPointIn2342 data container from and into a
    //!         scan2342 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScanPoint();

}; //Scan2342SerializedSize2342

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
