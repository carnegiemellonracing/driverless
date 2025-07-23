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
//! \date Aug 10, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/misc/SdkExceptions.hpp>

#include <cstddef> // for std::streamsize

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Base for classes used to export data into an IDC container.
//!
//! An exporter serializes a data container into a binary data block of a data type.
//------------------------------------------------------------------------------
class ExporterBase
{
private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::ExporterBase";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ExporterBase() = default;

    //========================================
    //! \brief Copy constructor.
    //!
    //! Copying is not allowed!
    //----------------------------------------
    ExporterBase(const ExporterBase&) = delete;

    //========================================
    //! \brief Assignment operator.
    //!
    //! Assignment is not allowed!
    //----------------------------------------
    ExporterBase& operator=(const ExporterBase&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~ExporterBase() = default;

public:
    //========================================
    //! \brief Get the data type to be exported into an IDC container.
    //!
    //! \return the data type this exporter can handle.
    //----------------------------------------
    virtual DataTypeId getDataType() const = 0;

    //========================================
    //! \brief Get the size in bytes that the object occupies when being serialized.
    //!
    //! \param[in] c  Object to get the serialized size for.
    //! \return  the number of bytes used for serialization.
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const = 0;

public:
    //========================================
    //! \brief Convert the data container to a serializable format and write it to the given stream (serialization).
    //!
    //! \param[in, out] os      Output data stream
    //! \param[in]      c       Data container to be serialized.
    //! \return \c True if serialization succeeds, \c false otherwise.
    //!
    //! \note This method has to be called from outside for serialization.
    //----------------------------------------
    virtual bool serialize(std::ostream& outStream, const DataContainerBase& c) const = 0;

    //========================================
    //! \brief Serialize data container to data package.
    //!
    //! \param[in] dataContainer    Instance of deserialized DataContainerBase.
    //! \return Either an instance of serialized IdcDataPackage if successful or otherwise nullptr.
    //----------------------------------------
    virtual IdcDataPackagePtr serializeToPackage(const DataContainerPtr& dataContainer) const;

    //========================================
    //! \brief Serialize data container to data package.
    //!
    //! \param[in] dataContainer    Instance of deserialized DataContainerBase.
    //! \return Either an instance of serialized IdcDataPackage if successful or otherwise nullptr.
    //----------------------------------------
    virtual IdcDataPackagePtr serializeToPackage(const DataContainerBase& dataContainer) const;

protected:
    //========================================
    //! \brief Get the data type to be exported into an IDC container.
    //!
    //! \return the data type this exporter can handle.
    //----------------------------------------
    virtual IdcDataPackagePtr createPackage() const;

}; // ExporterBase

//==============================================================================
//! \brief Template base class for all exporter specializations.
//!
//! A typed exporter serializes a data container into a binary data block of the templated data type.
//!
//! \tparam ClassContainerType  Which data container this exporter serializes.
//! \tparam serializationType   The data type id of the binary serialization.
//------------------------------------------------------------------------------
template<class ClassContainerType, microvision::common::sdk::DataTypeId::DataType serializationType>
class TypedExporter : public ExporterBase
{
public:
    //========================================
    //! \brief Get the data type ID of this specialized exporter.
    //!
    //! \return The data type ID of the serialization.
    //----------------------------------------
    DataTypeId getDataType() const final { return serializationType; }
}; // TypedExporter<>

//==============================================================================
//! \brief Template definition for Exporter specializations.
//!
//! An exporter serializes a data container into a binary data block of a data type.
//!
//! \tparam ClassContainerType  Which data container this exporter serializes.
//! \tparam serializationType The data type of the binary serialization.
//------------------------------------------------------------------------------
template<class ClassContainerType, microvision::common::sdk::DataTypeId::DataType serializationType>
class Exporter
{
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
