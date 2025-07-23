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
//! \date Sep 12, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/Configurable.hpp>
#include <microvision/common/sdk/config/Configuration.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainerBase.hpp>
#include <microvision/common/sdk/misc/SdkExceptions.hpp>

#include <boost/uuid/uuid.hpp>

#include <cstddef> // for std::streamsize

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class DataContainerBase;
class DataContainerListenerBase;

//==============================================================================
//! \brief Base for classes used to import data from an IDC container.
//!
//! An importer provides the functionality to deserialize a binary data block into a data container.
//------------------------------------------------------------------------------
class ImporterBase : public Configurable
{
public:
    class ImporterRegisterId;

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::ImporterBase";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static MICROVISION_SDK_API microvision::common::logging::LoggerSPtr m_logger;

public:
    //=================================================
    //! \brief Default constructor.
    //-------------------------------------------------
    ImporterBase() = default;

    //=================================================
    //! \brief Deleted copy constructor.
    //!
    //! Copying is not allowed!
    //-------------------------------------------------
    ImporterBase(const ImporterBase&) = delete;

    //=================================================
    //! \brief Deleted assignment operator.
    //!
    //! Assignment is not allowed!
    //-------------------------------------------------
    ImporterBase& operator=(const ImporterBase&) = delete;

    //=================================================
    //! \brief Default destructor.
    //-------------------------------------------------
    virtual ~ImporterBase() = default;

public: // implements Configurable
    //========================================
    //! \brief Get type of configuration to match with.
    //!
    //! This is a human readable unique string name of the configuration used to address it in code.
    //!
    //! \returns Configuration type.
    //! \note As default it will return an empty vector.
    //----------------------------------------
    const std::vector<std::string>& getConfigurationTypes() const override;

    //========================================
    //! \brief Get whether a configuration is mandatory for this Configurable.
    //! \return \c true if a configuration is mandatory for this Configurable,
    //!         \c false otherwise.
    //----------------------------------------
    bool isConfigurationMandatory() const override;

public:
    //=================================================
    //! \brief Get the data type to be imported from an IDC container.
    //!
    //! \return the data type this importer can handle.
    //-------------------------------------------------
    virtual DataTypeId getDataType() const = 0;

    //=================================================
    //! \brief Get the size in bytes that the object occupies when being serialized.
    //!
    //! \param[in] dataContainer  Object to get the serialized size for.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //! \return  the number of bytes used for serialization.
    //-------------------------------------------------
    virtual std::streamsize getSerializedSize(const DataContainerBase& dataContainer,
                                              const ConfigurationPtr& configuration = nullptr) const = 0;

    //=================================================
    //! \brief Read data from the given stream and fill the given data container (deserialization).
    //!
    //! \param[in, out] inputStream     Input data stream
    //! \param[out]     dataContainer   Output container defining the target type (might include conversion).
    //! \param[in]      dataHeader      Metadata prepended to each idc data block.
    //! \param[in]      configuration   (Optional) Configuration context for import. Default set as nullptr.
    //! \return \c True if deserialization succeeds, \c false otherwise.
    //!
    //! \note This method has to be called from outside for deserialization.
    //-------------------------------------------------
    virtual bool deserialize(std::istream& inputStream,
                             DataContainerBase& dataContainer,
                             const IdcDataHeader& dataHeader,
                             const ConfigurationPtr& configuration = nullptr) const = 0;

    //=================================================
    //! \brief Deserialize data package to given data container.
    //!
    //! \tparam    T                Target data container type.
    //! \param[in] dataPackage      Serialized IdcDataPackage input.
    //! \param[in] configuration    (Optional) Configuration context for import. Default set as nullptr.
    //! \return Either an instance of deserialized DataContainerBase if successful or otherwise nullptr.
    //!
    //! \note Be careful with the container type T - it may not match (or can not be casted) the content
    //!       of the data package and a nullptr will be returned!
    //-------------------------------------------------
    template<typename T>
    std::shared_ptr<T> deserializeTo(const IdcDataPackagePtr& dataPackage,
                                     const ConfigurationPtr& configuration = nullptr) const
    {
        const auto containerPtr = deserializeToContainer(dataPackage, configuration);
        if (containerPtr == nullptr)
        {
            return nullptr;
        }

        const auto targetPtr = std::dynamic_pointer_cast<T>(containerPtr);
        if (targetPtr == nullptr)
        {
            LOGWARNING(m_logger,
                       "Unable to cast deserialized container (from " << toHex(dataPackage->getHeader().getDataType())
                                                                      << ") to given target type!");
        }

        return targetPtr;
    }

    //=================================================
    //! \brief Deserialize data package into given data container.
    //!
    //! \param[in]      dataPackage     Serialized IdcDataPackage input (header with data type and binary data block).
    //! \param[in, out] dataContainer   Target container for the deserialization.
    //! \param[in]      configuration   (Optional) Configuration context for import. Default set as nullptr.
    //! \return Either an \c true if successful or otherwise \c false.
    //-------------------------------------------------
    bool deserializeToContainer(const IdcDataPackage& dataPackage,
                                DataContainerBase& dataContainer,
                                const ConfigurationPtr& configuration = nullptr) const;

    //=================================================
    //! \brief Deserialize data package to data container.
    //!
    //! \param[in] dataPackage      Serialized IdcDataPackage input (header with data type and binary data block).
    //! \param[in] configuration    (Optional) Configuration context for import. Default set as nullptr.
    //! \return Either an instance of deserialized DataContainerBase if successful or otherwise nullptr.
    //-------------------------------------------------
    virtual DataContainerPtr deserializeToContainer(const IdcDataPackagePtr& dataPackage,
                                                    const ConfigurationPtr& configuration = nullptr) const;

    //=================================================
    //! \brief Deserialize data package to data container.
    //!
    //! \param[in] dataPackage      Serialized IdcDataPackage input (header with data type and binary data block).
    //! \param[in] configuration    (Optional) Configuration context for import. Default set as nullptr.
    //! \return Either an instance of deserialized DataContainerBase if successful or otherwise \c nullptr.
    //-------------------------------------------------
    virtual DataContainerPtr deserializeToContainer(const IdcDataPackage& dataPackage,
                                                    const ConfigurationPtr& configuration = nullptr) const;

public:
    //=================================================
    //! \brief Create an instance of the target data container type.
    //!
    //! \return A pointer to an instance of the target data container type.
    //-------------------------------------------------
    virtual DataContainerPtr createContainer() const = 0;

public:
    //=================================================
    //! \brief Notify all registered listeners about the creation of a new object of the target type.
    //!
    //! \param[in] listener         Pointer to listener to be called.
    //! \param[in] container        Data container just created and filled with deserialized data.
    //! \param[in] configuration    (Optional) Configuration context for import. Default set as nullptr.
    //! \return \c True if successfully called the listener, \c false if not.
    //-------------------------------------------------
    virtual bool callListener(DataContainerListenerBase* listener,
                              const std::shared_ptr<const DataContainerBase> container,
                              const ConfigurationPtr& configuration = nullptr) const = 0;
}; // ImporterBase

//==============================================================================

//==============================================================================
//! \brief Key / value pair associating a data type identification with a function to create an importer.
//!
//! Used to register an importer for a given data type and data container combination.
//------------------------------------------------------------------------------
class ImporterBase::ImporterRegisterId
{
public:
    using Key = DataContainerBase::Key;

    using ImporterCreateFunction = std::function<ImporterBase*()>;

public:
    //=================================================
    //! \brief Constructor for registering importers for custom data containers.
    //!
    //! \param[in] importerId      Data type and container identification to be associated with the create function.
    //!                            For this one the key with a custom data container uuid is used.
    //! \param[in] importerCreate  Function to create the importer.
    //-------------------------------------------------
    ImporterRegisterId(const DataContainerBase::IdentificationKey& importerId,
                       const ImporterCreateFunction& importerCreate)
      : m_importerId{importerId}, m_importerCreate{importerCreate}
    {}

public:
    //=================================================
    //! \brief Get the key of this pair identifying the imported data type and the resulting data container.
    //!
    //! \return The identification of the associated importer.
    //-------------------------------------------------
    DataContainerBase::IdentificationKey getKey() const { return m_importerId; }

    //=================================================
    //! \brief Get the value of this pair which is the create function.
    //!
    //! \return the function used to create the importer.
    //-------------------------------------------------
    ImporterCreateFunction getValue() const { return m_importerCreate; }

private:
    DataContainerBase::IdentificationKey m_importerId;
    ImporterCreateFunction m_importerCreate;
}; // ImporterRegisterId

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
