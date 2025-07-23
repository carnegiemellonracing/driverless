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
//! \date Jun 24, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/special/2341/Scan2341.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This helper class is used to get the serialized size of all sub data containers used by the
//!        (de)serialization of Scan2341 data container from and into a scan2341 stream.
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED Scan2341SerializedSize2341
{
public:
    //========================================
    //! \brief Get the size of the serialization.
    //! \param[in] withScannerInfo  Either \c true to get scan size with scanner info,
    ///!                            otherwise \c false to get without.
    //! \return Number of bytes used by the (de)serialization of Scan2341 data container from and into a
    //!         Scan2341 stream.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScan(const bool withScannerInfo);

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScannerDirectionListIn2341 data container from and into a
    //!         Scan2341 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScannerDirectionList();

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScannerInfoIn2341 data container from and into a
    //!         Scan2341 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScannerInfo();

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScanPointRowIn2341 data container from and into a
    //!         scan2341 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScanPointRow();

    //========================================
    //! \brief Get the size of the serialization.
    //!
    //! \return Number of bytes used by the (de)serialization of ScanPointIn2341 data container from and into a
    //!         scan2341 stream.
    //! \note The parameter has no impact on the size calculation. The size is static.
    //----------------------------------------
    static std::streamsize getSerializedSizeOfScanPoint();

}; //Scan2341SerializedSize2341

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
