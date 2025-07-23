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
//! \date Mar 14, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessage64x0Base.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief FUSION SYSTEM/ECU Trace Messages
//!
//! Software modules that are deploying the FUSION SYSTEM for communication can sent trace messages
//! consisting of a character string. Trace Messages are distributed with four different data types dependent
//! on their priority
//------------------------------------------------------------------------------
class LogMessageDebug6430 final : public LogMessage64x0Base, public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;
    template<class ContainerType>
    friend class LogMessage64x0Exporter64x0Base;
    template<class ContainerType>
    friend class LogMessage64x0Importer64x0Base;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.logmessagedebug6430"};
    constexpr static const TraceLevel msgTraceLevel{TraceLevel::Debug};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    LogMessageDebug6430();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    virtual TraceLevel getTraceLevel() const override { return msgTraceLevel; }
}; // LogMessageDebug6430

//==============================================================================

bool operator==(const LogMessageDebug6430& lhs, const LogMessageDebug6430& rhs);
bool operator!=(const LogMessageDebug6430& lhs, const LogMessageDebug6430& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
