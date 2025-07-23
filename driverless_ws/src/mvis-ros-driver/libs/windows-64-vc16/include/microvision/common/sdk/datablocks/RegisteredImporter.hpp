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
//! \date Sep 8, 2017
///---------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ImporterBase.hpp>

#include <microvision/common/sdk/listener/DataContainerListener.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================
//! \brief Template class whose specializations will be derived from
//!        RegisteredImporter.
//------------------------------------------------------------------------------
template<class ContainerType, DataTypeId::DataType id>
class Importer
{
};

//==============================================================================
//! \brief Intermediate class between ImporterBase and Importer which
//!         provides registration to devices.
//------------------------------------------------------------------------------
template<class ContainerType, DataTypeId::DataType dataType>
class RegisteredImporter : public ImporterBase
{
public:
    //========================================
    //! \brief Create an instance of the importer.
    //!
    //! This is used to create the importer instance by the ImporterFactory.
    //!
    //! \return Shared pointer with the created importer.
    //----------------------------------------
    static std::shared_ptr<ImporterBase> createImporter()
    {
        return std::make_shared<Importer<ContainerType, dataType>>();
    }

    //========================================
    //! \brief Create an instance of the data container deserialized into by this importer.
    //!
    //! This is used to create a data container when importing.
    //!
    //! \return Shared pointer with the created data container.
    //-------------------------------------
    static std::shared_ptr<ContainerType> createContainerStatic() { return std::make_shared<ContainerType>(); }

    //========================================
    //! \brief Get the data type id of the serialization to be imported by this importer.
    //!
    //! \return Data type id of the container this importer creates.
    //----------------------------------------
    static DataTypeId getDataTypeStatic() { return static_cast<DataTypeId>(dataType); }

public:
    //========================================
    //! \brief No copy construction.
    //!
    //! Importers are meant to be stateless single instances.
    //----------------------------------------
    RegisteredImporter(const RegisteredImporter&) = delete;

    //========================================
    //! \brief No assignment operator.
    //!
    //! Importers are meant to be stateless single instances.
    //----------------------------------------
    RegisteredImporter& operator=(const RegisteredImporter&) = delete;

public:
    //========================================
    //! \brief Get the idc data type of the serialization imported by this importer.
    //!
    //! \return Data type id of the serialization this importer reads into a new data container.
    //----------------------------------------
    DataTypeId getDataType() const final { return getDataTypeStatic(); }

    //========================================
    //! \brief Create an instance of the data container filled by this importer.
    //!
    //! This is used to create a data container when importing.
    //!
    //! \return Shared pointer with the created data container.
    //----------------------------------------
    std::shared_ptr<DataContainerBase> createContainer() const final { return createContainerStatic(); }

public:
    //=================================================
    //! \brief Notify all registered listeners about the creation of a new object of the target type.
    //!
    //! \param[in] listener         Pointer to listener to be called.
    //! \param[in] container        Data container just created and filled with deserialized data.
    //! \param[in] configuration    (Optional) Configuration context for import. Default set as nullptr.
    //! \return \c True if successfully called the listener, \c false if not.
    //-------------------------------------------------
    bool callListener(DataContainerListenerBase* listener,
                      const std::shared_ptr<const DataContainerBase> container,
                      const ConfigurationPtr& configuration = nullptr) const override

    {
        // check if listener can handle this data container type
        auto dclImpl = dynamic_cast<DataContainerListener<ContainerType, dataType>*>(listener);
        if (dclImpl)
        {
            auto cPtr = std::dynamic_pointer_cast<const ContainerType>(container);
            if (cPtr)
            {
                dclImpl->onData(cPtr, configuration);
                return true;
            }
        }
        else
        {
            // might be a general data container -> try that
            auto gdclImpl = dynamic_cast<GeneralDataContainerListener<ContainerType>*>(listener);
            if (gdclImpl)
            {
                auto cPtr = std::dynamic_pointer_cast<const ContainerType>(container);
                if (cPtr)
                {
                    gdclImpl->onData(cPtr, configuration);
                    return true;
                }
            }
        }

        // check if custom data container uuid
        const auto uuid = findUuid<ContainerType>();
        if (uuid.is_nil())
        {
            // for custom data container types it would be ok to not find an importer here - but not for standard types!
            LOGWARNING(m_logger,
                       "No matching special or general listener found for the importer of idc data type "
                           << dataType << " (dynamic cast to container type failed)!");
        }

        return false;
    }

protected:
    //========================================
    //! \brief Protected constructor calling base.
    //!
    //! No need to create a RegisteredImporter directly.
    //----------------------------------------
    RegisteredImporter() : ImporterBase() {}

private:
    microvision::common::logging::LoggerSPtr m_logger{
        microvision::common::logging::LogManager::getInstance().createLogger(
            "microvision::common::sdk::RegisteredImporter")};
}; // class RegisteredImporter

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
