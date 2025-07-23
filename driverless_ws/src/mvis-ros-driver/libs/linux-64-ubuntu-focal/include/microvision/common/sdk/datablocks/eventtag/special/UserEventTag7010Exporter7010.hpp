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
//! \date Mar 15, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/eventtag/special/UserEventTag7010.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<UserEventTag7010, DataTypeId::DataType_UserEventTag7010>
  : public virtual TypedExporter<UserEventTag7010, DataTypeId::DataType_UserEventTag7010>
{
public:
    static constexpr std::streamsize serializedBaseSize{45 + UserEventTag7010::nbOfReserved};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // UserEventTag7010Exporter

//==============================================================================

using UserEventTag7010Exporter7010 = Exporter<UserEventTag7010, DataTypeId::DataType_UserEventTag7010>;

//==============================================================================

template<>
void writeBE<UserEventTag7010::TagClass>(std::ostream& os, const UserEventTag7010::TagClass& tc);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
