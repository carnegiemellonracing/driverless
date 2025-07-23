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

#include <microvision/common/sdk/io/DataPackageStreamWriter.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Extension interface for the StreamWriterFactory.
//!
//! Inherit this interface to implement a DataPackageStreamWriter for a new data source.
//!
//! \example IdcStreamWriterFactoryExtension
//------------------------------------------------------------------------------
class StreamWriterFactoryExtension
{
public:
    virtual ~StreamWriterFactoryExtension() = default;

public:
    //========================================
    //! \brief Creates a DataPackageStreamWriter from destination Uri.
    //! \param[in] path Valid Uri at destination system.
    //! \return Either an instance of DataPackageStreamWriter if supported or otherwise nullptr.
    //----------------------------------------
    virtual DataPackageStreamWriterPtr createPackageWriterFromUri(const Uri& path) const = 0;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
