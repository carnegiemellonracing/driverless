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
//! \date Sep 27, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/extension/StreamWriterFactory.hpp>
#include <microvision/common/sdk/extension/StreamReaderFactory.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>
#include <microvision/common/sdk/listener/IdcDataPackageListener.hpp>
#include <microvision/common/sdk/misc/IdcUriHelper.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Helper class to read an IDC file and inform listeners.
//------------------------------------------------------------------------------
class IdcFileInput final
{
public:
    //========================================
    //! \brief Creates a file device with the IDC file reader.
    //! \param[in] filePath  Valid file path of system.
    //----------------------------------------
    explicit IdcFileInput(const std::string& filePath) : m_reader{}
    {
        const Uri fileUri = IdcUriHelper::createIdcFormatFileUri(filePath);
        m_reader          = StreamReaderFactory::getInstance().createIdcPackageReaderFromUri(fileUri);
    }

public:
    //========================================
    //!\brief Register a DataListener to this device.
    //!
    //! Each time a message has been received by
    //! this object, the registered listener will be
    //! called which are listening to the received message
    //! type (DataType).
    //!
    //!\param[in] listener  Listener to be registered.
    //!
    //!\warning The methodology of this method has changed
    //!         the user do not need to register DataListeners for
    //!         each DataType individually.
    //----------------------------------------
    void registerDataContainerListener(const IdcEmitter::DataContainerListenerPtr& listener)
    {
        if (this->m_reader)
        {
            this->m_reader->registerDataContainerListener(listener);
        }
    }

    //========================================
    //!\brief Unregister a listener.
    //!\param[in] listener  Address of the listener object to be unregistered.
    //!\return Result of the operation.
    //----------------------------------------
    void unregisterDataContainerListener(const IdcEmitter::DataContainerListenerPtr& listener)
    {
        if (this->m_reader)
        {
            this->m_reader->unregisterDataContainerListener(listener);
        }
    }

    //========================================
    //!\brief Register a data package listener to this device.
    //!
    //! Each time a message has been received by the
    //! this object, the registered package listener will be
    //! call which are listening to the received message
    //! type (DataType).
    //!
    //!\param[in] listener  Package listener to be registered.
    //----------------------------------------
    void registerIdcDataPackageListener(const IdcEmitter::IdcDataPackageListenerPtr& listener)
    {
        if (this->m_reader)
        {
            this->m_reader->registerIdcDataPackageListener(listener);
        }
    }

    //========================================
    //!\brief Unregister a package listener.
    //!\param[in] listener  Address of the package listener object to be unregistered.
    //----------------------------------------
    void unregisterIdcDataPackageListener(const IdcEmitter::IdcDataPackageListenerPtr& listener)
    {
        if (this->m_reader)
        {
            this->m_reader->unregisterIdcDataPackageListener(listener);
        }
    }

public: // getter
    //========================================
    //! \brief Gets the device which provides the functionality to notify data.
    //! \return Either instance of IdcDataPackageStreamReader or if not set nullptr.
    //----------------------------------------
    const IdcDataPackageStreamReaderPtr& getReader() const { return this->m_reader; }

public: // setter
    //========================================
    //! \brief Sets the device which provides the functionality to notify data.
    //! \note The ownership of the pointer will overtake by the IdcFileInput.
    //! \param[in] reader  Either instance of IdcDataPackageStreamReader or nullptr to disable the reader.
    //----------------------------------------
    void setDevice(IdcDataPackageStreamReaderPtr&& reader) { this->m_reader = std::move(reader); }

public:
    //========================================
    //! \brief Tries to open the IDC file.
    //! \return Either \c true if IDC file is accessible or otherwise \c false.
    //!
    //! \note This checks if the file has a frame index. Broken/incomplete idc files have to be repaired
    //!       or opened without the help of this wrapper class. See idcRepair tool.
    //----------------------------------------
    bool open()
    {
        if (!this->m_reader || !this->m_reader->open())
        {
            return false;
        }

        // check if file has a frame index and trailer
        if (!this->m_reader->getFrameIndex() && !this->m_reader->getTrailer())
        {
            // would not help to have an invalid file open as simple idcInputFile wrapper
            // tools shall not use this wrapper
            this->m_reader->close();
            return false;
        }

        return true;
    }

    //========================================
    //! \brief Checks whether the IDC file is still accessible.
    //! \return Either \c true if IDC file is accessible or otherwise \c false.
    //----------------------------------------
    bool isOpen() const
    {
        if (this->m_reader)
        {
            return !this->m_reader->isBad();
        }
        return false;
    }

    //========================================
    //! \brief Checks whether the IDC file is still accessible and not EOF.
    //! \return Either \c true if IDC file is accessible and not EOF or otherwise \c false.
    //----------------------------------------
    bool isGood() const
    {
        if (this->m_reader)
        {
            return this->m_reader->isGood();
        }
        return false;
    }

    //========================================
    //! \brief Checks if the stream is not accessible or is unrecoverable or EOF.
    //! \return Either \c true if the resource is in bad or EOF condition, otherwise \c false.
    //----------------------------------------
    bool isEof() const
    {
        if (this->m_reader)
        {
            return this->m_reader->isEof();
        }
        return false;
    }

    //========================================
    //! \brief Released the IDC file resource.
    //----------------------------------------
    void close()
    {
        if (this->m_reader)
        {
            this->m_reader->close();
        }
    }

public:
    //========================================
    //! \brief Notify all registered Streamers and Package listeners.
    //! \param[in] dataPackage  Received IdcDataPackage by reader.
    //----------------------------------------
    bool notifyIdcDataPackageListeners(const IdcDataPackagePtr& dataPackage)
    {
        if (this->m_reader)
        {
            return this->m_reader->notifyIdcDataPackageListeners(dataPackage);
        }
        return false;
    }

    //========================================
    //! \brief Notify all registered container listeners to the received type of DataContainer.
    //!
    //! Use tryGetNextDataContainers(DataContainerMap& dataContainers) to get data containers.
    //!
    //! \param[in] dataContainers  All deserialized data container.
    //----------------------------------------
    bool notifyDataContainerListeners(const IdcDataPackagePtr& dataPackage)
    {
        if (this->m_reader)
        {
            return this->m_reader->notifyDataContainerListeners(dataPackage);
        }
        return false;
    }

    bool notifyDataContainerListeners(const IdcEmitter::DataContainerImporterContextList& dataContainers)
    {
        if (this->m_reader)
        {
            return this->m_reader->notifyDataContainerListeners(dataContainers);
        }
        return false;
    }

    bool notifyDataContainerListeners(const DataContainerPtr& dataContainer, const IdcEmitter::ImporterPtr& importer)
    {
        if (this->m_reader)
        {
            return this->m_reader->notifyDataContainerListeners(dataContainer, importer);
        }
        return false;
    }

public:
    //========================================
    //! \brief Utility method running through the IDC file and notifying all registered
    //!        streamers / listeners without time synchronisation
    //! \return The number of processed messages \c 0 if the IdcFile is not open/empty
    //!         or no listeners / streamers are registered.
    //----------------------------------------
    uint32_t loopAndNotify()
    {
        if (this->m_reader)
        {
            return this->m_reader->loopAndNotify();
        }
        return 0;
    }

    //========================================
    //! \brief Added all deserialization of the data package into data container by listener(s).
    //! \note The parameter dataContainers will only additive changed.
    //! \param[in] dataPackage      Read data package.
    //! \param[out] dataContainers  All conversions of the \a dataPackage.
    //! \return \c True if deserialization added to the vector, otherwise \c false.
    //----------------------------------------
    bool tryGetDataContainers(const IdcDataPackagePtr& dataPackage,
                              IdcEmitter::DataContainerImporterContextList& dataContainers)
    {
        if (this->m_reader)
        {
            return this->m_reader->tryGetDataContainers(dataPackage, dataContainers);
        }
        return false;
    }

    //========================================
    //! \brief Added all deserialization of the next data package into data container by listener(s).
    //! \note The parameter dataContainers will only additive changed.
    //! \param[out] dataContainers  All conversions of the next data package.
    //! \return \c True if deserialization added to the vector, otherwise \c false.
    //----------------------------------------
    bool tryGetNextDataContainers(IdcEmitter::DataContainerImporterContextList& dataContainers)
    {
        if (this->m_reader)
        {
            return this->m_reader->tryGetNextDataContainers(dataContainers);
        }
        return false;
    }

private:
    //========================================
    //! \brief Wrapped file device pointer.
    //----------------------------------------
    IdcDataPackageStreamReaderPtr m_reader;
}; // namespace sdk

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
