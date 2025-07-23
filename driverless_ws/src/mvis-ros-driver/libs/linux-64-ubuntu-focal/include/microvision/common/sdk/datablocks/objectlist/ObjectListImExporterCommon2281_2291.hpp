//==============================================================================
//! \file
//!
//! \brief Get serialized size for object types 0x2281/0x2291 from/to general object container.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 9, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/objectlist/ObjectList.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ObjectListImExporterCommon2281_2291
{
protected:
    // Coordinate system flags.
    static constexpr uint8_t flagsMiddleRearAxisISO70000{0x00};
    static constexpr uint8_t flagsMiddleFrontAxis{0x01};

    //========================================
    //!\brief get size in bytes of serialized data
    //!\param[in]      c       Input container.
    //!\return size in bytes
    //----------------------------------------
    static std::streamsize getSerializedSize(const DataContainerBase& c);

    //========================================
    //!\brief get size in bytes needed to serialize of a single general object
    //!       in the context of a data container 0x2281 or 0x2291
    //!\param[in]      c       Object.
    //!\return size in bytes
    //----------------------------------------
    static std::streamsize getSerializedSize(const Object& object);
}; // ObjectListImExporterCommon2281_2291

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
