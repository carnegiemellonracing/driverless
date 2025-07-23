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
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/marker/special/MarkerList6820.hpp>
#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Special exporter (0x6820) for MarkerList6820.
//!
//! This exporter provides the means to export a MarkerList6820 as a 0x6820
//! serialization.
//------------------------------------------------------------------------------
template<>
class Exporter<MarkerList6820, DataTypeId::DataType_Marker6820>
  : public TypedExporter<MarkerList6820, DataTypeId::DataType_Marker6820>
{
public:
    Exporter() : TypedExporter() {}
    Exporter(const Exporter&) = delete;
    Exporter& operator=(const Exporter&) = delete;

public:
    //========================================
    //! \brief Get size in bytes of the MarkerList6820 as serialization.
    //! \param[in] c  A reference to the MarkerList6820 object
    //!               the serialized size shall be calculated for
    //!               in form of a base class reference.
    //! \throws ContainerMismatch  if \a c is not actually a MarkerList6820.
    //! \return Size of the serialization of \a c in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    //========================================
    //! \brief Serialize a MarkerList6820 C++ object to the
    //!        output stream \a os.
    //! \param[in, out] os  The output stream the marker list
    //!                     shall be serialized to.
    //! \param[in]      c    A reference to the MarkerList6820 object
    //!                      to be serialized in form of a base
    //!                      class reference.
    //! \throws ContainerMismatch  if \a c is not actually a MarkerList6820.
    //! \return \c True if the serialization was successful, \c false otherwise.
    //----------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

private:
    //========================================
    //! \brief Serialize a string \a targetString to \a os.
    //!
    //! The length of the string is serialized before the string content as
    //! a uint16_t.

    //! \param[in,out] os            Output stream, the length of the string and the
    //!                              string is serialized to.
    //! \param[in]     sourceString  The string to be serialized.
    //----------------------------------------
    static void serializeString(std::ostream& os, const std::string& sourceString);
}; // MarkerList6820Exporter6820

//==============================================================================

using MarkerList6820Exporter6820 = Exporter<MarkerList6820, DataTypeId::DataType_Marker6820>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
