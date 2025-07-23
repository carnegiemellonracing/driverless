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
//! \date Feb 11, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/commands/Command2010.hpp>
#include <microvision/common/sdk/ImporterProvider.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

// Importer for command 2010 to special data container 2010
template<>
class Importer<Command2010, DataTypeId::DataType_Command2010>
  : public RegisteredImporter<Command2010, DataTypeId::DataType_Command2010>
{
public:
    Importer();
    Importer(const Importer&) = delete;
    Importer& operator=(const Importer&) = delete;

public: // implements ImporterBase
    //=================================================
    //! \brief Get the size in bytes that the object occupies when being serialized.
    //!
    //! \param[in] dataContainer  Object to get the serialized size for.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //! \return  the number of bytes used for serialization.
    //-------------------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& dataContainer,
                                      const ConfigurationPtr& configuration = nullptr) const override;

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
    bool deserialize(std::istream& inputStream,
                     DataContainerBase& dataContainer,
                     const IdcDataHeader& dataHeader,
                     const ConfigurationPtr& configuration = nullptr) const override;

public: // RegisteredImporter
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
                      const ConfigurationPtr& configuration = nullptr) const override;

    const ImporterProvider<CommandCBase>& getImporterProvider() const { return m_importerProvider; }
    ImporterProvider<CommandCBase>& getImporterProvider() { return m_importerProvider; }

private:
    bool deserializeSpecial(Command2010* container, const IdcDataHeader& dh) const;

    bool callCommandListener(DataContainerListenerBase* l, const DataContainerBase* const c) const;
    CommandId readCommandId(const char* const buffer) const;

private:
    static constexpr const char* loggerId = "microvision::common::sdk::Command2010Importer2010";
    static microvision::common::logging::LoggerSPtr logger;

private:
    mutable ImporterProvider<CommandCBase> m_importerProvider;
}; // Command2010Importer2010

//==============================================================================

using Command2010Importer2010 = Importer<Command2010, DataTypeId::DataType_Command2010>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
