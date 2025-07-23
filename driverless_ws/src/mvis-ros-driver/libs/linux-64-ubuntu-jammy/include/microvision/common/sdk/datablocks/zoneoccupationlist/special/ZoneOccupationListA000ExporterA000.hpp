//==============================================================================
//! \file
//!
//! \brief Special exporter for ZoneOccupationListA000 serialization.
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

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/ZoneOccupationListA000.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Special exporter (0xA000) for ZoneOccupationListA000
//!
//! This exporter provides the means to export a ZoneOccupationListA000 as a 0xA000
//! serialization.
//------------------------------------------------------------------------------
template<>
class Exporter<ZoneOccupationListA000, DataTypeId::DataType_ZoneOccupationListA000>
  : public TypedExporter<ZoneOccupationListA000, DataTypeId::DataType_ZoneOccupationListA000>
{
public:
    //========================================
    //! \brief Get size in bytes of the ZoneOccupationListA000 as serialization.
    //! \param[in] c  A reference to the ZoneOccupationListA000 object
    //!               the serialized size shall be calculated for
    //!               in form of a base class reference.
    //! \throws ContainerMismatch  if \a c is not actually a ZoneOccupationListA000.
    //! \return Size of the serialization of \a c in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    //========================================
    //! \brief Serialize a ZoneOccupationListA000 C++ object to the
    //!        output stream \a os.
    //! \param[in, out] os  The output stream the marker list
    //!                     shall be serialized to.
    //! \param[in]      c    A reference to the ZoneOccupationListA000 object
    //!                      to be serialized in form of a base
    //!                      class reference.
    //! \throws ContainerMismatch  if \a c is not actually a ZoneOccupationListA000.
    //! \return Either \c true if the serialization was successful, \c false otherwise.
    //----------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

private:
    //========================================
    //! \brief Serialize a zone definition to the output stream.
    //!
    //! Writes all properties of the zone definition to the stream in binary format.
    //!
    //! \param[in,out] os              Output stream to serialize the zone definition to.
    //! \param[in]     zoneDefinition  The zone definition to be serialized.
    //! \return Either \c true if the serialization was successful, \c false otherwise.
    //----------------------------------------
    static bool serialize(std::ostream& os, const ZoneDefinitionInA000& zoneDefinition);

    //========================================
    //! \brief Serialize a rigid transformation to the output stream.
    //!
    //! Writes position and orientation values to the stream in binary format.
    //!
    //! \param[in,out] os    Output stream to serialize the transformation to.
    //! \param[in]     pose  The rigid transformation to be serialized.
    //! \return Either \c true if the serialization was successful, \c false otherwise.
    //----------------------------------------
    static bool serialize(std::ostream& os, const RigidTransformationInA000& pose);

    //========================================
    //! \brief Serialize a string \a s to \a os.
    //!
    //! The length of the string is serialized before the string content as
    //! a uint32_t.
    //!
    //! \param[in,out] os  Output stream, the length of the string and the
    //!                    string is serialized to.
    //! \param[in]     s   The string to be serialized.
    //! \return Either \c true if the serialization was successful, \c false otherwise.
    //----------------------------------------
    static bool serialize(std::ostream& os, const std::string& s);

    //========================================
    //! \brief Serialize a vector of ints \a v to \a os.
    //!
    //! The size of the vector is serialized before the string content as
    //! a uint32_t.
    //!
    //! \param[in,out] os          Output stream, the size of the vector and the
    //!                            vector is serialized to.
    //! \param[in]     intVector   The vector to be serialized.
    //! \return Either \c true if the serialization was successful, \c false otherwise.
    //----------------------------------------
    static bool serialize(std::ostream& os, const std::vector<int32_t>& intVector);

    //========================================
    //! \brief Serialize a vector of zone states to the output stream.
    //!
    //! Writes the number of states followed by each zone state in binary format.
    //!
    //! \param[in,out] os            Output stream to serialize the zone states to.
    //! \param[in]     stateVector   The vector of zone states to be serialized.
    //! \return Either \c true if the serialization was successful, \c false otherwise.
    //----------------------------------------
    static bool serialize(std::ostream& os, const std::vector<ZoneStateInA000>& stateVector);

}; // ZoneOccupationListA0000ExporterA000

//==============================================================================

using ZoneOccupationListA000ExporterA000
    = Exporter<ZoneOccupationListA000, DataTypeId::DataType_ZoneOccupationListA000>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
