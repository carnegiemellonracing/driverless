//==============================================================================
//! \file
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PerceptionDataInfo.hpp>

#include <microvision/common/sdk/datablocks/marker/special/MarkerIn6820.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief A container to hold a list of Markers.
//!
//! Markers provide a simple mean to store generic data for a visualization.
//! The main purpose is meant to be debugging.
//!
//! \note Markers(List) supposed to replace LogPolygon(Lists)
//------------------------------------------------------------------------------
class MarkerList6820 final : public SpecializedDataContainer
{
public:
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Vector of the markers used in MarkerList6820.
    //----------------------------------------
    using MarkerVector = std::vector<MarkerIn6820>;

    //========================================
    //! \brief A vector of strings used to define
    //!        a marker's namespace.
    //----------------------------------------
    using MarkerNamespace = MarkerIn6820::MarkerNamespace;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.markerlist6820"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    MarkerList6820() = default;

    //========================================
    //! \brief Destrutor.
    //----------------------------------------
    ~MarkerList6820() override = default;

    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    //========================================
    //! \brief Get the PerceptionDataInfo.
    //! \return Return the PerceptionDataInfo.
    //----------------------------------------
    const PerceptionDataInfo& getPerceptionDataInfo() const { return m_dataInfo; }

    //========================================
    //! \brief Get the marker list's measurement time stamp.
    //! \return Return the marker list's measurement time stamp.
    //----------------------------------------
    NtpTime getMeasurementTimestamp() const { return m_measurementTimestamp; }

    //========================================
    //! \brief Get the marker namespace prefix.
    //!
    //! Depending on the marker's setting, this
    //! marker namespace prefix will be used
    //! as a prefix for the namespace given in
    //! the marker itself or will be ignored
    //! and replaced by the marker's namespace.
    //!
    //! \return Returns a const reference to
    //!         the marker namespace prefix.
    //----------------------------------------
    const MarkerNamespace& getMarkerNamespacePrefix() const { return m_markerNamespacePrefix; }

    //========================================
    //! \brief Get the marker namespace prefix.
    //!
    //! Depending on the marker's setting, this
    //! marker namespace prefix will be used
    //! as a prefix for the namespace given in
    //! the marker itself or will be ignored
    //! and replaced by the marker's namespace.
    //!
    //! \return Returns a reference to the marker
    //!         namespace prefix.
    //----------------------------------------
    MarkerNamespace& getMarkerNamespacePrefix() { return m_markerNamespacePrefix; }

    //========================================
    //! \brief Get the marker vector.
    //! \return Return a const reference to the marker vector.
    //----------------------------------------
    const MarkerVector& getMarkerVector() const { return m_markerVector; }

    //========================================
    //! \brief Get the marker vector.
    //! \return Return a non-const reference to the marker vector.
    //----------------------------------------
    MarkerVector& getMarkerVector() { return m_markerVector; }

public: // setter
    //========================================
    //! \brief Set a new PerceptionDataInfo.
    //! \param[in] newDataInfo  The new PerceptionDataInfo.
    //----------------------------------------
    void setPerceptionDataInfo(const PerceptionDataInfo& newDataInfo) { m_dataInfo = newDataInfo; }

    //========================================
    //! \brief Set a new measurement timestamp.
    //! \param[in] newMeasurementTimestamp  The marker list's
    //!                                     new measurement timestamp
    //----------------------------------------
    void setMeasurementTimestamp(const NtpTime newMeasurementTimestamp)
    {
        m_measurementTimestamp = newMeasurementTimestamp;
    }

    //========================================
    //! \brief Set the marker namespace prefix.
    //! \param[in] newPrefix  The new marker namespace prefix.
    //!
    //! Depending on the marker's setting, this
    //! marker namespace prefix will be used
    //! as a prefix for the namespace given in
    //! the marker itself or will be ignored
    //! and replaced by the marker's namespace.
    //----------------------------------------
    void setMarkerNamespacePrefix(const MarkerNamespace& newPrefix) { m_markerNamespacePrefix = newPrefix; }

    //========================================
    //! \brief Set the marker's namespace prefix.
    //! \param[in,out] newPrefix  The new marker's namespace prefix.
    //!
    //! Depending on the marker's setting, this
    //! marker namespace prefix will be used
    //! as a prefix for the namespace given in
    //! the marker itself or will be ignored
    //! and replaced by the marker's namespace.
    //----------------------------------------
    void setMarkerNamespacePrefix(MarkerNamespace&& newPrefix) { m_markerNamespacePrefix = std::move(newPrefix); }

    //========================================
    //! \brief Set a new marker vector.
    //! \param[in] newMarkerVector  The new marker vector to be copied
    //!                             into the MarkerList6820.
    //----------------------------------------
    void setMarkerVector(const MarkerVector& newMarkerVector) { m_markerVector = newMarkerVector; }

    //========================================
    //! \brief Set a new marker vector.
    //! \param[in,out] newMarkerVector  The new marker vector to be moved.
    //!                                 into the MarkerList6820.
    //----------------------------------------
    void setMarkerVector(MarkerVector&& newMarkerVector) { m_markerVector = std::move(newMarkerVector); }

private:
    //========================================
    //! \brief Some meta data for the marker data.
    //----------------------------------------
    PerceptionDataInfo m_dataInfo;

    //========================================
    //! \brief Timestamp of the measurement.
    //----------------------------------------
    NtpTime m_measurementTimestamp;

    //========================================
    //! \brief The marker's namespace prefix.
    //!
    //! The namespace prefix is represented as
    //!  a vector of strings to be read.
    //!
    //! String[0]::String[1]::String[2]...
    //!
    //! Depending on the marker's setting, this
    //! marker namespace prefix will be used
    //! as a prefix for the namespace given in
    //! the marker itself or will be ignored
    //! and replaced by the marker's namespace.
    //----------------------------------------
    MarkerNamespace m_markerNamespacePrefix{};

    //========================================
    //! \brief The vector of the contained markers.
    //----------------------------------------
    MarkerVector m_markerVector;
}; // MarkerList6820

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const MarkerList6820& lhs, const MarkerList6820& rhs);

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const MarkerList6820& lhs, const MarkerList6820& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
