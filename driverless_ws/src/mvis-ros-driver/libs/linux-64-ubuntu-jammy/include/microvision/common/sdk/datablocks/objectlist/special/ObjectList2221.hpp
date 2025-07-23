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
//! \date Jan 17, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/objectlist/special/ObjectIn2221.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of objects in the format of the LUX3 firmware
//!
//! Object data available from MVIS LUX laser scanners (not available for MVIS LUX prototypes).
//! Each data block starts with a header followed by the object list. Each object has a list of contour points.
//! The Sigma values are calculated by Kalman filter by taking into account the object age.
//------------------------------------------------------------------------------
class ObjectList2221 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ObjectVector = std::vector<ObjectIn2221>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.objectlist2221"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectList2221() : SpecializedDataContainer() {}
    ObjectList2221(const NtpTime scanStartTimeStamp)
      : SpecializedDataContainer(), m_scanStartTimestamp{scanStartTimeStamp}
    {}

    virtual ~ObjectList2221() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    void addObject(const ObjectIn2221& newObj) { m_objects.push_back(newObj); }
    const ObjectIn2221* findObject(const uint16_t id) const;

public: // getter
    NtpTime getScanStartTimestamp() const { return m_scanStartTimestamp; }
    uint16_t getNbOfObjects() const { return static_cast<uint16_t>(m_objects.size()); }

    const ObjectVector& getObjects() const { return m_objects; }
    ObjectVector& getObjects() { return m_objects; }

public: // setter
    void setScanStartTimeStamp(const NtpTime newScanStartTimeStamp) { m_scanStartTimestamp = newScanStartTimeStamp; }
    void setObjects(const ObjectVector& objects) { m_objects = objects; }

protected:
    NtpTime m_scanStartTimestamp{0}; //!< Timestamp of the first scan these objects are updated with.
    ObjectVector m_objects{};
}; // ObjectList2221

//==============================================================================

inline bool operator==(const ObjectList2221& lhs, const ObjectList2221& rhs)
{
    return (lhs.getScanStartTimestamp() == rhs.getScanStartTimestamp()) && (lhs.getObjects() == rhs.getObjects());
}

inline bool operator!=(const ObjectList2221& lhs, const ObjectList2221& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
