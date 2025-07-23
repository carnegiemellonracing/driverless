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
//! \date Mar 26, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class EventMarker7001 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.EventMarker7001"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    EventMarker7001();
    EventMarker7001(const uint16_t type, const std::string& body, const std::string& author);
    EventMarker7001(const uint16_t type, const char* const body, const char* const author);

    virtual ~EventMarker7001() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // Getter
    uint16_t getType() const { return m_type; }
    const std::string& getBody() const { return m_body; }
    const std::string& getAuthor() const { return m_author; }

public: // Setter
    void setType(const uint16_t type) { m_type = type; }
    void setBody(const std::string& b) { m_body = b; }
    void setAuthor(const std::string& a) { m_author = a; }

protected:
    uint16_t m_type; //!< Type of event
    std::string m_body; //!< String containing the event.
    std::string m_author; //!< The authorship of the event
}; // EventMarker7001

//==============================================================================

bool operator==(const EventMarker7001& lhs, const EventMarker7001& rhs);
bool operator!=(const EventMarker7001& lhs, const EventMarker7001& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
