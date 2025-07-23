//==============================================================================
//! \file
//!
//! \brief Provides the base functionality to read/write BAG file.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 05, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/WinCompatibility.hpp>
#include <microvision/common/sdk/misc/SharedBuffer.hpp>
#include <microvision/common/sdk/io/StreamBase.hpp>

#include <microvision/common/sdk/io/bag/package/BagConnectionRecordPackage.hpp>
#include <microvision/common/sdk/io/bag/package/BagChunkInfoRecordPackage.hpp>
#include <microvision/common/sdk/io/bag/package/BagMessageRecordPackage.hpp>
#include <microvision/common/sdk/io/bag/package/BagHeaderRecordPackage.hpp>
#include <microvision/common/sdk/io/bag/package/BagIndexRecordPackage.hpp>
#include <microvision/common/sdk/io/bag/package/BagChunkRecordPackage.hpp>

#include <functional>
#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Provides the base functionality to read/write BAG file.
//! \extends StreamBase
//------------------------------------------------------------------------------
class BagStreamBase : public virtual StreamBase
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using StreamBase::open;

public: // constants
    //========================================
    //! \brief Name of the BAG format for the Uri.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string bagFormatName;

    //========================================
    //! \brief Name of the BAG raw schema for the Uri.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string bagRecordSchemaName;

    //========================================
    //! \brief File name extension for BAG files.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string bagFilenameExtension;

    //========================================
    //! \brief Identifier for BAG file format version 2.0
    //----------------------------------------
    static MICROVISION_SDK_API const std::string version20;

    //========================================
    //! \brief BAG file format delimiter character.
    //----------------------------------------
    static constexpr char fileFormatDelimiter{0x0a};

public: // utillity functions
    //========================================
    //! \brief Either \c true if the uri represent a file or
    //!         addressless uri of format BAG, otherwise \c false.
    //! \param[in] path  Uri to check.
    //! \return Either \c true if the format of uri is BAG, otherwise \c false.
    //----------------------------------------
    static bool hasBagFormat(const Uri& path);

    //========================================
    //! \brief Create a file Uri as BAG format from file path.
    //! \param[in] filePath  Valid file path of system.
    //! \return Created Uri of protocol file and BAG format.
    //----------------------------------------
    static Uri createBagFileUri(const std::string& filePath);

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::BagStreamBase";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor which requires a uri of format BAG.
    //! \param[in] path  Valid Uri of source system.
    //----------------------------------------
    explicit BagStreamBase(const Uri& path);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    BagStreamBase(BagStreamBase&& reader);

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    BagStreamBase(const BagStreamBase&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~BagStreamBase() override;

public:
    //========================================
    //! \brief Move assignment operator.
    //----------------------------------------
    BagStreamBase& operator=(BagStreamBase&& reader);

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    BagStreamBase& operator=(const BagStreamBase& reader) = delete;

public:
    //========================================
    //! \brief Get BAG format version
    //! \return Either BagFormatVersion::Version20 if stream is supported
    //!         or otherwise BagFormatVersion::Invalid.
    //----------------------------------------
    BagFormatVersion getVersion() const;

public: // implements StreamBase
    //========================================
    //! \brief Get pointer to configuration which is used to define/filter/prepare input.
    //! \return Pointer to an instance of Configuration.
    //----------------------------------------
    ConfigurationPtr getConfiguration() const override;

    //========================================
    //! \brief Set pointer to configuration which is used to define/filter/prepare input.
    //!
    //! \param[in] configuration  Pointer to an instance of Configuration.
    //! \return Either \c true if the configuration is supported by implementation or otherwise \c false.
    //! \note If the configuration is not supported by implementation it will not change the current value.
    //!       However, if \a configuration is \c nullptr the configuration of NetworkInterface will be reset.
    //----------------------------------------
    bool setConfiguration(const ConfigurationPtr& configuration) override;

    //========================================
    //! \brief Get the source Uri.
    //! \return Describing source Uri of stream.
    //----------------------------------------
    const Uri& getUri() const override;

    //========================================
    //! \brief Checks if the stream is accessible
    //! \attention That does not check if the stream is in failed state because of
    //!             the possibility to start a new read process with clearing the flags.
    //! \return Either \c true if the resource is in good condition, otherwise \c false.
    //----------------------------------------
    bool isGood() const override;

    //========================================
    //! \brief Checks if the stream is not accessible or is unrecoverable.
    //! \return Either \c true if the resource is in bad condition, otherwise \c false.
    //----------------------------------------
    bool isBad() const override;

    //========================================
    //! \brief Checks if the stream is not accessible or is unrecoverable or EOF.
    //! \return Either \c true if the resource is in bad or EOF condition, otherwise \c false.
    //----------------------------------------
    bool isEof() const override;

    //========================================
    //! \brief Release resources and stream ownership.
    //! \returns Get stream ownership back.
    //----------------------------------------
    IoStreamPtr release() override;

public: // implement Configurable
    //========================================
    //! \brief Get supported types of configuration.
    //!
    //! Configuration type is a human readable unique string name of the configuration
    //! used to address it in code.
    //!
    //! \return All supported configuration types.
    //----------------------------------------
    const std::vector<std::string>& getConfigurationTypes() const override;

    //========================================
    //! \brief Get whether a configuration is mandatory for this Configurable.
    //! \return \c true if a configuration is mandatory for this Configurable,
    //!         \c false otherwise.
    //----------------------------------------
    bool isConfigurationMandatory() const override;

protected:
    //========================================
    //! \brief Set badbit on stream.
    //----------------------------------------
    virtual void setStreamInvalid();

    //========================================
    //! \brief Check whether path is valid format/schema.
    //! \return Either \c true if path format/schema is valid, otherwise \c false.
    //----------------------------------------
    virtual bool checkPath() const;

    //========================================
    //! \brief Seek back at starting position if action returns false.
    //! \param[in] action  Functional pointer to read/write function which has to be executed.
    //! \returns Result of action or \c false if seek back failed.
    //----------------------------------------
    bool tryActionAndSeekbackOnFailure(const std::function<bool()>& action);

    //========================================
    //! \brief Read version at start of stream
    //! \returns Either \c true if version can be readed
    //!          and is supported or otherwise \c false.
    //----------------------------------------
    bool readVersion();

    //========================================
    //! \brief Read record in raw BAG package
    //! \param[in] package  Raw BAG package
    //! \return Either \c true if successful or otherwise \c false.
    //----------------------------------------
    bool readRecord(BagRecordPackagePtr& package);

protected:
    //========================================
    //! \brief BAG format version
    //----------------------------------------
    std::string m_version;

    //========================================
    //! \brief Stream handle or nullptr if not accessible.
    //----------------------------------------
    IoStreamPtr m_stream;

    //========================================
    //! \brief Stream source Uri.
    //----------------------------------------
    Uri m_path;

    //========================================
    //! \brief Mutex used for a threadsafe implementation.
    //----------------------------------------
    mutable Mutex m_mutex;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
