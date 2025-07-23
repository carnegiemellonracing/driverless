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
//! \date Apr 18, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/datablocks/stateofoperation/special/StateOfOperation9110.hpp>
#include <microvision/common/sdk/datablocks/stateofoperation/special/StateOfOperation9111.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief State of operation at a given time
//!
//! Special data types:
//! \ref microvision::common::sdk::StateOfOperation9111
//! \ref microvision::common::sdk::StateOfOperation9110 (deprecated)
//------------------------------------------------------------------------------
class StateOfOperation final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const StateOfOperation&, const StateOfOperation&);

public:
    using State           = StateOfOperation9111::State; //!< The state type from StateOfOperation9111.
    using ArrayOfReserved = StateOfOperation9111::ArrayOfReserved; //!< The array for the reserved bytes.

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.stateofoperation"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    StateOfOperation() = default;

    //========================================
    //! \brief Default copy constructor.
    //----------------------------------------
    StateOfOperation(const StateOfOperation& rhs) = default;

    //========================================
    //! \brief Default assignment constructor.
    //----------------------------------------
    StateOfOperation& operator=(const StateOfOperation& rhs) = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~StateOfOperation() override = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    //========================================
    //! \brief Get the timestamp of the state of operation.
    //! \return The timestamp.
    //----------------------------------------
    const Timestamp& getTimestamp() const { return m_delegate.getTimestamp(); }

    //========================================
    //! \brief Get the state type of the state of operation.
    //! \return The state type.
    //----------------------------------------
    State getState() const { return m_delegate.getState(); }

    //========================================
    //! \brief Get the active flag of the state of operation.
    //! \return The active flag.
    //----------------------------------------
    bool getActiveState() const { return m_delegate.getActiveState(); }

    //========================================
    //! \brief Get the driver active flag of the state of operation.
    //! \return The driver active flag.
    //----------------------------------------
    bool getDriverActive() const { return m_delegate.getDriverActive(); }

    //========================================
    //! \brief Get the reserved bytes of the state of operation.
    //! \return The reserved bytes.
    //----------------------------------------
    const ArrayOfReserved& getReserved() const { return m_delegate.getReserved(); }

public: // setter
    //========================================
    //! \brief Sets the timestamp of the state of operation.
    //! \param[in] timestamp  The new timestamp.
    //----------------------------------------
    void setTimestamp(const Timestamp& timestamp) { m_delegate.setTimestamp(timestamp); }

    //========================================
    //! \brief Sets the state type of the state of operation.
    //! \param[in] state  The new state type.
    //----------------------------------------
    void setState(const State type) { m_delegate.setState(type); }

    //========================================
    //! \brief Sets the flag for the active state of the state of operation.
    //! \param[in] active  The new active state.
    //----------------------------------------
    void setActiveState(const bool active) { m_delegate.setActiveState(active); }

    //========================================
    //! \brief Sets the flag for the driver active state of the state of operation.
    //! \param[in] active  The new driver active state.
    //----------------------------------------
    void setDriverActive(const bool driverActive) { m_delegate.setDriverActive(driverActive); }

protected:
    //========================================
    //! \brief The used specialization.
    //! \note The old specialization is no longer used.
    // ----------------------------------------
    StateOfOperation9111 m_delegate{};
}; // StateOfOperation

//==============================================================================
// Specializations for operators.
//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator==(const StateOfOperation& lhs, const StateOfOperation& rhs)
{
    return (lhs.m_delegate == rhs.m_delegate);
}

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const StateOfOperation& lhs, const StateOfOperation& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
