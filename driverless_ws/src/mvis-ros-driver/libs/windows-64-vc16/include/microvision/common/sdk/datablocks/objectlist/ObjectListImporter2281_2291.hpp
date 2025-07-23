//==============================================================================
//! \file
//!
//! \brief Im- or exports object type 0x2281/0x2291 from/to general object container.
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

#include <microvision/common/sdk/datablocks/objectlist/ObjectListImExporterCommon2281_2291.hpp>
#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ObjectListImporter2281_2291 : public ObjectListImExporterCommon2281_2291
{
public:
    //========================================
    //!\brief create data from byte stream (deserialization)
    //!\param[in, out] is      Input data stream
    //!\param[in]      c       Input container.
    //!\param[in]      header  idc dataHeader
    //!\return \c True if serialization succeed, else: \c false
    //!\note This method is to be called from outside for deserialization.
    //----------------------------------------
    static bool deserialize(std::istream& is, DataContainerBase& c, const IdcDataHeader& header);

private:
    // Deserialize the general object in context of a data container type 0x2281/0x2291.
    static bool deserialize(std::istream& is, Object& object);
}; // ObjectListImporter2281_2291

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
