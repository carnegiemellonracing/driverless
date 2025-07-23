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
//! \date Jun 5, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/DataPackageStreamReader.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Extension interface for the StreamReaderFactory.
//!
//! Inherit this interface to implement a DataPackageStreamReader for new data source.
//!
//! \example IdcStreamReaderFactoryExtension.
//------------------------------------------------------------------------------
class StreamReaderFactoryExtension
{
public:
    virtual ~StreamReaderFactoryExtension() = default;

public:
    //========================================
    //! \brief Creates a DataPackageStreamReader from source Uri.
    //! \param[in] path  Valid Uri of source system.
    //! \return Either instance of DataPackageStreamReader if supported or otherwise nullptr.
    //----------------------------------------
    virtual DataPackageStreamReaderPtr createPackageReaderFromUri(const Uri& path) const = 0;
}; // StreamReaderFactoryExtension

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
