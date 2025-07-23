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
//! \date Feb 04, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief State of operation at a given time
//------------------------------------------------------------------------------
class StateOfOperation9111 : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const uint8_t nbOfReserved = 4; //!< The number of reserved bytes in this state of operation.

    using ArrayOfReserved = std::array<uint32_t, nbOfReserved>; //!< The array for the reserved bytes.

public:
    //========================================
    //! \brief The state type of the state of operation.
    //----------------------------------------
    enum class State : uint8_t
    {
        NotReady    = 0,
        Ready       = 1,
        Driving     = 5,
        Standstill  = 6,
        Takeoverreq = 10,
        Fallback    = 11,
        Error       = 15
    };

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.stateofoperation9111"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    StateOfOperation9111();

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~StateOfOperation9111() override;

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
    const Timestamp& getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief Get the state type of the state of operation.
    //! \return The state type.
    //----------------------------------------
    State getState() const { return m_state; }

    //========================================
    //! \brief Get the active flag of the state of operation.
    //! \return The active flag.
    //----------------------------------------
    bool getActiveState() const { return m_active; }

    //========================================
    //! \brief Get the driver active flag of the state of operation.
    //! \return The driver active flag.
    //----------------------------------------
    bool getDriverActive() const { return m_driverActive; }

    //========================================
    //! \brief Get the reserved bytes of the state of operation.
    //! \return The reserved bytes.
    //----------------------------------------
    const ArrayOfReserved& getReserved() const { return m_reserved; }

public: // setter
    //========================================
    //! \brief Sets the timestamp of the state of operation.
    //! \param[in] timestamp  The new timestamp.
    //----------------------------------------
    void setTimestamp(const Timestamp& timestamp) { m_timestamp = timestamp; }

    //========================================
    //! \brief Sets the state type of the state of operation.
    //! \param[in] state  The new state type.
    //----------------------------------------
    void setState(const State state) { m_state = state; }

    //========================================
    //! \brief Sets the flag for the active state of the state of operation.
    //! \param[in] active  The new active state.
    //----------------------------------------
    void setActiveState(const bool active) { m_active = active; }

    //========================================
    //! \brief Sets the flag for the driver active state of the state of operation.
    //! \param[in] active  The new driver active state.
    //----------------------------------------
    void setDriverActive(const bool active) { m_driverActive = active; }

protected:
    Timestamp m_timestamp{}; //!< the timestamp of this state of operation.
    State m_state{State::NotReady}; //!< The state of this state of operation.
    bool m_active{false}; //!< The active flag of this state of operation.
    bool m_driverActive{false}; //!< The driver active flag of this state of operation.

private:
    ArrayOfReserved m_reserved{{0U, 0U, 0U, 0U}}; //!< The reserved bytes of the state of operation.
}; // StateOfOperation9111

//==============================================================================
// Specializations for operators.
//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const StateOfOperation9111& lhs, const StateOfOperation9111& rhs);

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator!=(const StateOfOperation9111& lhs, const StateOfOperation9111& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
