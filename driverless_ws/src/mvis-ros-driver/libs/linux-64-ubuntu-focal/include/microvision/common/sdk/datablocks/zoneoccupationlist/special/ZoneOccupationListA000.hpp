//==============================================================================
//! \file
//!
//! \brief Data container to store zone occupation list information.
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

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/RigidTransformationInA000.hpp>
#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/ZoneDefinitionInA000.hpp>
#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/ZoneStateInA000.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data container for zone occupation list information.
//------------------------------------------------------------------------------
class ZoneOccupationListA000 final : public SpecializedDataContainer
{
public:
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Maximum number of zones supported.
    //----------------------------------------
    static constexpr uint32_t maxNumberOfZones{64};

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.zoneoccupationlistA000"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Calculate hash value for zone definitions.
    //! \param[in] zoneDefinitions Vector of zone definitions.
    //! \return Hash value for the zone definitions.
    //----------------------------------------
    static uint32_t calculateZoneDefinitionsHash(const std::vector<ZoneDefinitionInA000>& zoneDefinitions);

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ZoneOccupationListA000() = default;

public: // DataContainerBase implementation
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Check if the zone occupation list is valid.
    //!
    //! The following condition has to be met to make a ZoneOccupationListA000 valid:
    //! - The number of zone states is equal to the number of zone definitions.
    //! - All ZoneDefinitionInA000 IDs have to be unique.
    //! - For each ZoneDefinitionInA000
    //!   - The number of vertex x coordinates is equal to the number of y coordinates.
    //!   - The number of vertices is 0 or at least 3. Less than 3 vertices do not define
    //!     an area but just a line.
    //!   - The z extrusion maximum has to be greater than the minimum to define a volume.
    //!
    //! \return Either \c true if the list is valid, \c false otherwise.
    //----------------------------------------
    bool isValid() const;

public: // getters
    //========================================
    //! \brief Get measurement ID.
    //! \return ID of the corresponding measurement.
    //----------------------------------------
    uint32_t getMeasurementId() const;

    //========================================
    //! \brief Get measurement time in seconds.
    //! \return Timestamp of the corresponding measurement, full seconds.
    //----------------------------------------
    uint64_t getMeasurementTimeSeconds() const;

    //========================================
    //! \brief Get measurement time fractions.
    //! \return Timestamp of the corresponding measurement, fraction of a second.
    //----------------------------------------
    uint64_t getMeasurementTimeFractions() const;

    //========================================
    //! \brief Get sensor mounting pose.
    //! \return Pose of the sensor within the safety zones' coordinate frame.
    //----------------------------------------
    const RigidTransformationInA000& getSensorMountingPose() const;

    //========================================
    //! \brief Get zone states.
    //! \return Vector of zone states.
    //----------------------------------------
    const std::vector<ZoneStateInA000>& getZoneStates() const;

    //========================================
    //! \brief Get zone definitions.
    //! \return Vector of zone definitions.
    //----------------------------------------
    const std::vector<ZoneDefinitionInA000>& getZoneDefinitions() const;

    //========================================
    //! \brief Get zone definitions hash.
    //! \return Hash value computed for zone definitions.
    //----------------------------------------
    uint32_t getZoneDefinitionsHash() const;

public: // setters
    //========================================
    //! \brief Set measurement ID.
    //! \param[in] measurementId  ID of the corresponding measurement.
    //----------------------------------------
    void setMeasurementId(const uint32_t measurementId);

    //========================================
    //! \brief Set measurement time in seconds.
    //! \param[in] measurementTimeSeconds  Timestamp of the corresponding measurement, full seconds.
    //----------------------------------------
    void setMeasurementTimeSeconds(const uint64_t measurementTimeSeconds);

    //========================================
    //! \brief Set measurement time fractions.
    //! \param[in] measurementTimeFractions  Timestamp of the corresponding measurement, fraction of a second.
    //----------------------------------------
    void setMeasurementTimeFractions(const uint64_t measurementTimeFractions);

    //========================================
    //! \brief Set sensor mounting pose.
    //! \param[in] sensorMountingPose  Pose of the sensor within the safety zones' coordinate frame.
    //----------------------------------------
    void setSensorMountingPose(const RigidTransformationInA000& sensorMountingPose);

    //========================================
    //! \brief Set zone states.
    //! \param[in] zoneStates  Vector of zone states.
    //! \return Either \c true if the number of zone states is not greater than #maxNumberOfZones.
    //!         \c false otherwise.
    //----------------------------------------
    bool setZoneStates(const std::vector<ZoneStateInA000>& zoneStates);

    //========================================
    //! \brief Set zone definitions by copying.
    //! \param[in] zoneDefinitions  Vector of zone definitions.
    //! \return Either \c true if the number of zone definition is not greater than #maxNumberOfZones.
    //!         \c false otherwise.
    //----------------------------------------
    bool setZoneDefinitions(const std::vector<ZoneDefinitionInA000>& zoneDefinitions);

    //========================================
    //! \brief Set zone definitions by moving.
    //! \param[in] zoneDefinitions  Vector of zone definitions.
    //! \return Either \c true if the number of zone definition is not greater than #maxNumberOfZones.
    //!         \c false otherwise.
    //----------------------------------------
    bool setZoneDefinitions(std::vector<ZoneDefinitionInA000>&& zoneDefinitions);

    //========================================
    //! \brief Append a zone definition by copying.
    //! \param[in] zoneDefinition  Zone definition to append.
    //! \return Either \c true if the number of zone definition is smaller than #maxNumberOfZones.
    //!         \c false otherwise.
    //----------------------------------------
    bool appendZoneDefinition(const ZoneDefinitionInA000& zoneDefinition);

    //========================================
    //! \brief Append a zone definition by moving.
    //! \param[in] zoneDefinition  Zone definition to append.
    //! \return Either \c true if the number of zone definition is smaller than #maxNumberOfZones.
    //!         \c false otherwise.
    //----------------------------------------
    bool appendZoneDefinition(ZoneDefinitionInA000&& zoneDefinition);

    //========================================
    //! \brief Update a zone definition by copying.
    //! \param[in] indexOfZone     Index of the zone to update.
    //! \param[in] zoneDefinition  New zone definition.
    //! \return Either \c true if indexOfZone is a valid index into the vector of zone definitions.
    //!         \c false otherwise.
    //----------------------------------------
    bool updateZoneDefinition(const uint32_t indexOfZone, const ZoneDefinitionInA000& zoneDefinition);

    //========================================
    //! \brief Update a zone definition by moving.
    //! \param[in] indexOfZone     Index of the zone to update.
    //! \param[in] zoneDefinition  New zone definition.
    //! \return Either \c true if indexOfZone is a valid index into the vector of zone definitions.
    //!         \c false otherwise.
    //----------------------------------------
    bool updateZoneDefinition(const uint32_t indexOfZone, ZoneDefinitionInA000&& zoneDefinition);

    //========================================
    //! \brief Update zone vertices by copying.
    //! \param[in] indexOfZone    Index of the zone to update.
    //! \param[in] verticesXInMm  Vector of X coordinates in millimeters.
    //! \param[in] verticesYInMm  Vector of Y coordinates in millimeters.
    //! \return Either \c true if indexOfZone is a valid index into the vector of zone definitions
    //!         and the number of x and y positions are identical and less than or equal
    //!         to #ZoneDefinitionInA000::maxNumberOfVertices. \c false otherwise.
    //----------------------------------------
    bool updateZoneVertices(const uint32_t indexOfZone,
                            const std::vector<int32_t>& verticesXInMm,
                            const std::vector<int32_t>& verticesYInMm);

    //========================================
    //! \brief Update zone vertices by moving.
    //! \param[in] indexOfZone    Index of the zone to update.
    //! \param[in] verticesXInMm  Vector of X coordinates in millimeters.
    //! \param[in] verticesYInMm  Vector of Y coordinates in millimeters.
    //! \return Either \c true if indexOfZone is a valid index into the vector of zone definitions
    //!         and the number of x and y positions are identical and less than or equal
    //!         to #ZoneDefinitionInA000::maxNumberOfVertices. \c false otherwise.
    //----------------------------------------
    bool updateZoneVertices(const uint32_t indexOfZone,
                            std::vector<int32_t>&& verticesXInMm,
                            std::vector<int32_t>&& verticesYInMm);

    //========================================
    //! \brief Update zone pose.
    //! \param[in] indexOfZone  Index of the zone to update.
    //! \param[in] pose         New rigid transformation representing the zone pose.
    //! \return Either \c true if indexOfZone is a valid index into the vector of zone definitions.
    //!         \c false otherwise.
    //----------------------------------------
    bool updateZonePose(const uint32_t indexOfZone, const RigidTransformationInA000& pose);

private:
    uint32_t m_measurementId{0}; //!< ID of the corresponding measurement.

    uint64_t m_measurementTimeSeconds{0}; //!< Timestamp of the corresponding measurement, full seconds.
    uint64_t m_measurementTimeFractions{0}; //!< Timestamp of the corresponding measurement, fraction of a second.

    RigidTransformationInA000 m_sensorMountingPose; //!< Pose of the sensor within the safety zones' coordinate frame

    std::vector<ZoneStateInA000> m_zoneStates;
    std::vector<ZoneDefinitionInA000> m_zoneDefinitions;
    uint32_t m_zoneDefinitionsHash{0}; //!< Hash value computed for m_zoneDefinitions.
}; // ZoneOccupationListA000

//==============================================================================
//! \brief Equality comparison operator for ZoneOccupationListA000.
//! \param[in] lhs  Left-hand side operand.
//! \param[in] rhs  Right-hand side operand.
//! \return Either \c true if all properties are equal, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const ZoneOccupationListA000& lhs, const ZoneOccupationListA000& rhs);

//==============================================================================
//! \brief Inequality comparison operator for ZoneOccupationListA000.
//! \param[in] lhs  Left-hand side operand.
//! \param[in] rhs  Right-hand side operand.
//! \return Either \c true if any property differs, \c false if all are equal.
//------------------------------------------------------------------------------
inline bool operator!=(const ZoneOccupationListA000& lhs, const ZoneOccupationListA000& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
