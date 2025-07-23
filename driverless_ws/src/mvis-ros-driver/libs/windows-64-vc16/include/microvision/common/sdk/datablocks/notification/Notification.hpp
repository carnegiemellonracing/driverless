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
//! \date Jun 19, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/notification/special/Notification2030.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Notification
//!
//! A notification is an error code that can be emitted from a Device or
//! an Interface to inform its registered message handlers about problems.
//!
//! \ref microvision:common:sdk:Notification2030
//------------------------------------------------------------------------------
class Notification final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.notification"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Notification();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~Notification() override = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Get the mnemonic of the notification.
    //----------------------------------------
    const std::string& getMnemonic() const { return m_delegate.getMnemonic(); }

    //========================================
    //! \brief Get the severity of the notification.
    //----------------------------------------
    Notification2030::TraceLevel getSeverity() const { return m_delegate.getSeverity(); }

public:
    //========================================
    //! \brief Set the mnemonic of the notification.
    //----------------------------------------
    void setMnemonic(const std::string& newMnemonic) { m_delegate.setMnemonic(newMnemonic); }

    //========================================
    //! \brief Set the mnemonic of the notification.
    //----------------------------------------
    void setMnemonic(std::string&& newMnemonic) { m_delegate.setMnemonic(newMnemonic); }

    //========================================
    //! \brief Set the severity of the notification.
    //----------------------------------------
    void setSeverity(Notification2030::TraceLevel newTraceLevel) { m_delegate.setSeverity(newTraceLevel); }

protected:
    Notification2030 m_delegate; // only possible specialization currently
}; // NotificationContainer

//==============================================================================

bool operator==(const Notification& lhs, const Notification& rhs);
bool operator!=(const Notification& lhs, const Notification& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
