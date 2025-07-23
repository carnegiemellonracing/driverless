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
//! \date May 23, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/eventmarker/special/EventMarker7001.hpp>
#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Event marker
//!
//! Special data type: \ref microvision::common::sdk::EventMarker7001
//------------------------------------------------------------------------------
class EventMarker final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.EventMarker"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    EventMarker() : DataContainerBase() {}

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~EventMarker() = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // Getter
    //========================================
    //!\brief Get the type of the event
    //!\return type of the event
    //----------------------------------------
    uint16_t getType() const { return m_delegate.getType(); }

    //========================================
    //!\brief Get the body of the event
    //!\return body of the event
    //----------------------------------------
    const std::string& getBody() const { return m_delegate.getBody(); }

    //========================================
    //!\brief Get the author of the event
    //!\return author of the event
    //----------------------------------------
    const std::string& getAuthor() const { return m_delegate.getAuthor(); }

public: // Setter
    //========================================
    //!\brief Set the type of the event
    //!\param[in] type  New type of the event
    //----------------------------------------
    void setType(const uint16_t type) { m_delegate.setType(type); }

    //========================================
    //!\brief Set the body of the event.
    //!\param[in] body  New body of the event
    //----------------------------------------
    void setBody(const std::string& body) { m_delegate.setBody(body); }

    //========================================
    //!\brief Set the author of the event
    //!\param[in] author  New author of the event
    //----------------------------------------
    void setAuthor(const std::string& author) { m_delegate.setAuthor(author); }

protected:
    EventMarker7001 m_delegate; // only possible specialization currently
}; // EventMarker

//==============================================================================

bool operator==(const EventMarker& lhs, const EventMarker& rhs);
bool operator!=(const EventMarker& lhs, const EventMarker& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
