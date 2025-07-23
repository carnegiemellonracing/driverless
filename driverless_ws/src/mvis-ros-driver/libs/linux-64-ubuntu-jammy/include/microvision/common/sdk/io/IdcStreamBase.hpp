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
//! \date Aug 19, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/StreamBase.hpp>
#include <microvision/common/sdk/datablocks/idctrailer/special/IdcTrailer6120.hpp>
#include <microvision/common/sdk/datablocks/frameindex/special/FrameIndex6130.hpp>
#include <microvision/common/sdk/datablocks/frameindex/special/FramingPolicyIn6130.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Interface for idc stream.
//!
//! Shared functionality for IdcDataPackageStreamReader and IdcDataPackageStreamWriter.
//------------------------------------------------------------------------------
class IdcStreamBase : public virtual StreamBase
{
public:
    //========================================
    //! \brief FrameIndex6130 shared pointer type.
    //----------------------------------------
    using FrameIndexPtr = std::shared_ptr<FrameIndex6130>;

    //========================================
    //! \brief IdcTrailer6120 shared pointer type.
    //----------------------------------------
    using IdcTrailerPtr = std::shared_ptr<IdcTrailer6120>;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcStreamBase() override = default;

public:
    //========================================
    //! \brief Gets a pointer to the frame index.
    //! \return Pointer of FrameIndex6130 if exists, otherwise nullptr.
    //----------------------------------------
    virtual FrameIndexPtr getFrameIndex() = 0;

    //========================================
    //! \brief Gets a pointer to the idc trailer.
    //! \return Pointer of IdcTrailer6120 if exists, otherwise nullptr.
    //----------------------------------------
    virtual IdcTrailerPtr getTrailer() = 0;

    //========================================
    //! \brief Move cursor to frame begin.
    //! \return Either \c true if successful, otherwise \c false.
    //----------------------------------------
    virtual bool seekFrame(const FrameIndexEntryIn6130& frame) = 0;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
