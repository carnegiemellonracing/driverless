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
//! \date Mar 16, 2018
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
class StateOfOperation9110 final : public SpecializedDataContainer
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
    //! \brief Operation modes.
    //----------------------------------------
    enum class Operation : uint8_t
    {
        Off     = 1,
        Standby = 2,
        Ready   = 3,
        Running = 4
    };

    //========================================
    //! \brief Driver Interrupt types.
    //----------------------------------------
    enum class DriverInterrupt : uint8_t
    {
        None     = 1,
        Cancel   = 2,
        Takeover = 3
    };

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.stateofoperation9110"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    StateOfOperation9110();

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~StateOfOperation9110() override;

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
    //! \brief Get the operating mode of the state of operation.
    //! \return The operating mode.
    //----------------------------------------
    Operation getOperation() const { return m_operation; }

    //========================================
    //! \brief Get the interrupt type of the state of operation.
    //! \return The interrupt type.
    //----------------------------------------
    DriverInterrupt getDriverInterrupt() const { return m_driverInterrupt; }

    //========================================
    //! \brief Get the takeOver flag of the state of operation.
    //! \return The takeOver flag.
    //----------------------------------------
    bool getTakeOverRequest() const { return m_takeOverRequest; }

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
    //! \brief Sets the operation type of the state of operation.
    //! \param[in] type  The new operation type.
    //----------------------------------------
    void setOperation(const Operation type) { m_operation = type; }

    //========================================
    //! \brief Sets the interrupt type of the state of operation.
    //! \param[in] type  The new interrupt type.
    //----------------------------------------
    void setDriverInterrupt(const DriverInterrupt type) { m_driverInterrupt = type; }

    //========================================
    //! \brief Sets the flag for the take over request of the state of operation.
    //! \param[in] request  The new take over request.
    //----------------------------------------
    void setTakeOverRequest(const bool request) { m_takeOverRequest = request; }

protected:
    Timestamp m_timestamp{}; //!< The timestamp of this state of operation.
    Operation m_operation{Operation::Off}; //!< The operating mode.
    DriverInterrupt m_driverInterrupt{DriverInterrupt::None}; //!< The interrupt type of this state of operation.
    bool m_takeOverRequest{false}; //!< The flag for the take over request.

private:
    ArrayOfReserved m_reserved{{0U, 0U, 0U, 0U}}; //!< The reserved bytes of the state of operation.
}; // StateOfOperation9110

//==============================================================================
// Specializations for operators.
//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const StateOfOperation9110& lhs, const StateOfOperation9110& rhs);

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator!=(const StateOfOperation9110& lhs, const StateOfOperation9110& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
