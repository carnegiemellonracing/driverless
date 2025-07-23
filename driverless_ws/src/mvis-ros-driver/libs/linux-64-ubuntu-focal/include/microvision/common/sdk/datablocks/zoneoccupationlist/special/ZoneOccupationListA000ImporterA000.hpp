//==============================================================================
//! \file
//!
//! \brief Special importer for ZoneOccupationListA000 deserialization.
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

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/ZoneOccupationListA000.hpp>
#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/zoneOccupationListA000Support.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Special importer (0xA000) for ZoneOccupationListA000
//!
//! This importer provides the means to import a ZoneOccupationListA000 from a 0xA000
//! serialization.
//------------------------------------------------------------------------------
template<>
class Importer<ZoneOccupationListA000, DataTypeId::DataType_ZoneOccupationListA000>
  : public RegisteredImporter<ZoneOccupationListA000, DataTypeId::DataType_ZoneOccupationListA000>
{
public:
    //========================================
    //! \brief Empty constructor calling base.
    //----------------------------------------
    Importer() : RegisteredImporter() {}

    //========================================
    //! \brief Copy construction is forbidden.
    //----------------------------------------
    Importer(const Importer&) = delete;

    //========================================
    //! \brief Assignment construction is forbidden.
    //----------------------------------------
    Importer& operator=(const Importer&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Importer() override = default;

public: // implements ImporterBase
    //========================================
    //! \brief Get the size in bytes that the object occupies when being serialized.
    //!
    //! \param[in] dataContainer  Object to get the serialized size for.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //! \return  the number of bytes used for serialization.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& dataContainer,
                                      const ConfigurationPtr& configuration = nullptr) const override;

    //========================================
    //! \brief Read data from the given stream and fill the given data container (deserialization).
    //!
    //! \param[in, out] inputStream     Input data stream
    //! \param[out]     dataContainer   Output container defining the target type (might include conversion).
    //! \param[in]      dataHeader      Metadata prepended to each idc data block.
    //! \param[in]      configuration   (Optional) Configuration context for import. Default set as nullptr.
    //! \return Either \c true if deserialization succeeds, \c false otherwise.
    //!
    //! \note This method has to be called from outside for deserialization.
    //----------------------------------------
    bool deserialize(std::istream& inputStream,
                     DataContainerBase& dataContainer,
                     const IdcDataHeader& dataHeader,
                     const ConfigurationPtr& configuration = nullptr) const override;

private:
    //========================================
    //! \brief Deserialize a string from an input stream.
    //!
    //! The length of the string is serialized before the string content as
    //! a uint16_t.
    //!
    //! \param[in,out] is  Input stream, the length of the following string
    //!                    and the string itself is read from.
    //! \param[out]    s   The string, where the result of the
    //!                    deserialization is stored.
    //! \return Either \c true if deserialization succeeds, \c false otherwise.
    //----------------------------------------
    static bool deserialize(std::istream& is, std::string& s);

    //========================================
    //! \brief Deserialize a vector of integers from an input stream.
    //!
    //! The size of the vector is read first, followed by the integer values.
    //!
    //! \param[in,out] is         Input stream to read the vector data from.
    //! \param[out]    intVector  The vector to store the deserialized integers.
    //! \return Either \c true if deserialization succeeds, \c false otherwise.
    //----------------------------------------
    static bool deserialize(std::istream& is, std::vector<int32_t>& intVector);

    //========================================
    //! \brief Deserialize a vector of zone states from an input stream.
    //!
    //! Reads the number of states followed by each zone state in binary format.
    //!
    //! \param[in,out] is           Input stream to read the zone states from.
    //! \param[out]    stateVector  The vector to store the deserialized zone states.
    //! \return Either \c true if deserialization succeeds, \c false otherwise.
    //----------------------------------------
    static bool deserialize(std::istream& is, std::vector<ZoneStateInA000>& stateVector);

    //========================================
    //! \brief Deserialize a rigid transformation from an input stream.
    //!
    //! Reads position and orientation values from the stream in binary format.
    //!
    //! \param[in,out] is    Input stream to read the transformation from.
    //! \param[out]    pose  The rigid transformation to be populated.
    //! \return Either \c true if deserialization succeeds, \c false otherwise.
    //----------------------------------------
    static bool deserialize(std::istream& is, RigidTransformationInA000& pose);

    //========================================
    //! \brief Deserialize a zone definition from an input stream.
    //!
    //! Reads all properties of the zone definition from the stream in binary format.
    //!
    //! \param[in,out] is              Input stream to read the zone definition from.
    //! \param[out]    zoneDefinition  The zone definition to be populated.
    //! \return Either \c true if deserialization succeeds, \c false otherwise.
    //----------------------------------------
    static bool deserialize(std::istream& is, ZoneDefinitionInA000& zoneDefinition);

    //========================================
    //! \brief Calculate the serialized size of a rigid transformation.
    //!
    //! \param[in] zone  The rigid transformation to calculate size for.
    //! \return The size in bytes that the object will occupy when serialized.
    //----------------------------------------
    static std::streamsize getSerializedSize(const RigidTransformationInA000& zone);

    //========================================
    //! \brief Calculate the serialized size of a zone definition.
    //!
    //! \param[in] zone  The zone definition to calculate size for.
    //! \return The size in bytes that the object will occupy when serialized.
    //----------------------------------------
    static std::streamsize getSerializedSize(const ZoneDefinitionInA000& zone);

}; //ZoneOccupationListA000ImporterA000

//==============================================================================

using ZoneOccupationListA000ImporterA000
    = Importer<ZoneOccupationListA000, DataTypeId::DataType_ZoneOccupationListA000>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
