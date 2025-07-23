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
//! \date Jun 11, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/IdcDataPackageStreamReader.hpp>

#include <functional>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Implements functionality to read IdcDataPackages from idc stream.
//!
//! Provides functions to read forward and backward in idc stream.
//! An idc stream is defined by IdcDataHeader.
//------------------------------------------------------------------------------
class IdcStreamReader : public IdcDataPackageStreamReader
{
public:
    //========================================
    //! \brief Data package pointer with related stream position.
    //----------------------------------------
    using IndexedDataPackagePtr = std::pair<std::streampos, IdcDataPackagePtr>;

    //========================================
    //! \brief Data container pointer with related stream position.
    //----------------------------------------
    using IndexedDataContainerPtr = std::pair<std::streampos, DataContainerPtr>;

    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using IdcDataPackageStreamReader::open;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::IdcStreamReader";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor which requires a uri of format IDC.
    //! \param[in] path  Valid Uri of source system.
    //----------------------------------------
    explicit IdcStreamReader(const Uri& path);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    IdcStreamReader(IdcStreamReader&& reader);

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    IdcStreamReader(const IdcStreamReader&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcStreamReader() override;

public:
    //========================================
    //! \brief Move assignment operator.
    //----------------------------------------
    IdcStreamReader& operator=(IdcStreamReader&& reader);

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    IdcStreamReader& operator=(const IdcStreamReader& reader) = delete;

public: // implements IdcDataPackageStreamReader
    //========================================
    //! \brief Peek first IdcDataHeader.
    //! \return First IdcDataHeader if found, otherwise nullptr.
    //----------------------------------------
    IdcDataHeaderPtr peekFirstIdcDataHeader() override;

    //========================================
    //! \brief Peek last IdcDataHeader.
    //! \return Last IdcDataHeader if found, otherwise nullptr.
    //----------------------------------------
    IdcDataHeaderPtr peekLastIdcDataHeader() override;

    //========================================
    //! \brief Peek next IdcDataHeader.
    //! \return Next IdcDataHeader if found, otherwise nullptr.
    //----------------------------------------
    IdcDataHeaderPtr peekNextIdcDataHeader() override;

    //========================================
    //! \brief Peek for MagicWord.
    //! \return \c True, if next bytes are the MagicWord, \c false otherwise.
    //----------------------------------------
    bool peekMagicWord();

    //========================================
    //! \brief Peek previous IdcDataHeader.
    //! \return Previous IdcDataHeader if found, otherwise nullptr.
    //----------------------------------------
    IdcDataHeaderPtr peekPreviousIdcDataHeader() override;

    //========================================
    //! \brief Read first IdcDataPackage.
    //! \note Will ignored IdcTrailer and FrameIndex.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    IdcDataPackagePtr readFirstIdcDataPackage() override;

    //========================================
    //! \brief Read last IdcDataPackage.
    //! \note Will ignored IdcTrailer and FrameIndex.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    IdcDataPackagePtr readLastIdcDataPackage() override;

    //========================================
    //! \brief Read next IdcDataPackage.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    IdcDataPackagePtr readNextIdcDataPackage() override;

    //========================================
    //! \brief Read previous IdcDataPackage.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    IdcDataPackagePtr readPreviousIdcDataPackage() override;

public: // implements DataPackageStreamReader
    //========================================
    //! \brief Skip the next X IdcDataPackage blocks.
    //! \param[in] packageCount  Count of packages too skip.
    //! \return \c True if successful, otherwise \c false.
    //----------------------------------------
    bool skipNextPackages(const uint32_t packageCount) override;

    //========================================
    //! \brief Skip the n previous IdcDataPackage blocks.
    //! \param[in] packageCount  Count of packages too skip.
    //! \return \c True if successful, otherwise \c false.
    //----------------------------------------
    bool skipPreviousPackages(const uint32_t packageCount) override;

public: //implements IdcStreamBase
    //========================================
    //! \brief Gets a pointer to the FrameIndex6130.
    //! \return Pointer of FrameIndex6130 if exists, otherwise nullptr.
    //----------------------------------------
    FrameIndexPtr getFrameIndex() override;

    //========================================
    //! \brief Gets a pointer to the idc trailer.
    //! \return Pointer of IdcTrailer6120 if exists, otherwise nullptr.
    //----------------------------------------
    IdcTrailerPtr getTrailer() override;

    //========================================
    //! \brief Seek cursor to frame begin.
    //! \param[in] frame  Seek to position of FrameIndexEntryIn6130.
    //! \return \c True if successful, otherwise \c false.
    //----------------------------------------
    bool seekFrame(const FrameIndexEntryIn6130& frame) override;

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
    //!             the possible to start a new read process with clearing the flags.
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
    //! \brief Request resource access and takeover of the stream ownership.
    //! \note Supports only the uri protocols UriProtocol::File and UriProtocol::NoAddr.
    //! \param[in] ioStream  Resource Stream handle.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream) override;

    //========================================
    //! \brief Release resources and stream ownership.
    //! \returns Get stream ownership back.
    //----------------------------------------
    IoStreamPtr release() override;

    //========================================
    //! \brief Seek the cursor position.
    //! \param[in] cursor  Target cursor position.
    //! \return Either \c true if possible, otherwise \c false.
    //----------------------------------------
    bool seek(const int64_t cursor) override;

    //========================================
    //! \brief Seek cursor to begin of stream.
    //! \return Either \c true if successful, otherwise \c false.
    //----------------------------------------
    bool seekBegin() override;

    //========================================
    //! \brief Seek cursor to end of stream.
    //! \return Either \c true if successful, otherwise \c false.
    //----------------------------------------
    bool seekEnd() override;

    //========================================
    //! \brief Get the current cursor position.
    //! \return Current cursor position or -1 for EOF.
    //----------------------------------------
    int64_t tell() override;

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

private:
    //========================================
    //! \brief Seek to the end and set the members m_indexFrame (FrameIndex),
    //!        m_idcTrailer (IdcTrailer) and m_last (DataPackage).
    //! \return \c True if last IdcDataPackage found, otherwise \c false.
    //----------------------------------------
    bool seekFindEnd();

    //========================================
    //! \brief Peek to the end and set the members m_indexFrame (FrameIndex),
    //!        m_idcTrailer (IdcTrailer) and m_last (DataPackage).
    //! \return \c True if last IdcDataPackage found, otherwise \c false.
    //----------------------------------------
    bool peekFindEnd();

    //========================================
    //! \brief Read IdcDataHeader at current position.
    //! \param[in] seekBackPosition  Position where to seek back.
    //! \return Either \c IdcDataHeader if readable or otherwise \c nullptr.
    //----------------------------------------
    IdcDataHeaderPtr peekDataHeader(const std::streampos seekBackPosition);

private:
    //========================================
    //!\brief File position and data container of frame index at the end of the file,
    //!       otherwise nullptr.
    //----------------------------------------
    IndexedDataContainerPtr m_indexFrame;

    //========================================
    //!\brief File position and data container of idc trailer at the end of the file,
    //!       otherwise nullptr.
    //----------------------------------------
    IndexedDataContainerPtr m_idcTrailer;

    //========================================
    //!\brief File position and data package at the start of the file, otherwise nullptr.
    //----------------------------------------
    IndexedDataPackagePtr m_first;

    //========================================
    //!\brief File position and data package at the end of the file, otherwise nullptr.
    //----------------------------------------
    IndexedDataPackagePtr m_last;

    //========================================
    //!\brief Stream handle or nullptr if not accessible.
    //----------------------------------------
    IoStreamPtr m_stream;

    //========================================
    //!\brief Stream source Uri.
    //----------------------------------------
    Uri m_path;

    //========================================
    //!\brief Mutex for ThreadSafe implementation.
    //----------------------------------------
    mutable Mutex m_mutex;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
