//==============================================================================
//! \file
//!
//! \brief Support functions for ZoneOccupationListA000 serialization and deserialization.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 29, 2025
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/ZoneOccupationListA000.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace zoneoccupationlistA000 {
//==============================================================================

//==============================================================================
//! \brief Calculate the serialized size of a ZoneOccupationListA000 object.
//! \param[in] zoneOccupationList  The zone occupation list to calculate size for.
//! \return The size in bytes that the object will occupy when serialized.
//------------------------------------------------------------------------------
std::streamsize getSerializedSize(const ZoneOccupationListA000& zoneOccupationList);

//==============================================================================
//! \brief Calculate the serialized size of a RigidTransformationInA000 object.
//! \param[in] pose  The rigid transformation to calculate size for.
//! \return The size in bytes that the object will occupy when serialized.
//------------------------------------------------------------------------------
uint32_t getSerializedSize(const RigidTransformationInA000& pose);

//==============================================================================
//! \brief Calculate the serialized size of a ZoneDefinitionInA000 object.
//! \param[in] zoneDefinition  The zone definition to calculate size for.
//! \return The size in bytes that the object will occupy when serialized.
//------------------------------------------------------------------------------
uint32_t getSerializedSize(const ZoneDefinitionInA000& zoneDefinition);

//==============================================================================
} // namespace zoneoccupationlistA000
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
