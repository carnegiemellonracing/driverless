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

#include <microvision/common/sdk/io/IdcDataPackageStreamWriter.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief idc stream writer is an implementation of IdcDataPackageStreamWriter.
//!
//! Provides functions to write idc data packages in idc stream.
//------------------------------------------------------------------------------
class IdcStreamWriter : public IdcDataPackageStreamWriter
{
private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::IdcStreamWriter";

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using IdcDataPackageStreamWriter::open;

public:
    //========================================
    //! \brief Default constructor which requires a Uri of format IDC.
    //! \param[in] path  Valid Uri of source system.
    //----------------------------------------
    explicit IdcStreamWriter(const Uri& path);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    IdcStreamWriter(IdcStreamWriter&& writer);

    //========================================
    //! \brief Copy constructor is disabled because of stream dependencie.
    //----------------------------------------
    IdcStreamWriter(const IdcStreamWriter&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcStreamWriter() override;

public:
    //========================================
    //! \brief Move assignment operator.
    //----------------------------------------
    IdcStreamWriter& operator=(IdcStreamWriter&& writer);

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    IdcStreamWriter& operator=(const IdcStreamWriter& writer) = delete;

public: // implements IdcDataPackageStreamWriter
    //========================================
    //! \brief Request resource access to append data if possible and takeover of the stream ownership.
    //! \note Supports only the uri protocols UriProtocol::File and UriProtocol::NoAddr.
    //! \param[in] ioStream     Resource stream handle.
    //! \param[in] append       \c true if the resource requested is in append mode, otherwise \c false.
    //! \param[in] policy       Defined where a frame starts.
    //! \note In append mode the frame index will fixed by reading the stream.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream, const bool appendMode, const FramingPolicyIn6130& policy) override;

public: // implements DataPackageStreamWriter
    //========================================
    //! \brief Write IdcDataPackage at stream position.
    //! \param[in/out] data  IdcDataPackage to write.
    //! \returns Either \c true if IdcDataPackage successful written, otherwise \c false.
    //----------------------------------------
    bool writePackage(DataPackage& dataPackage) override;

    //========================================
    //! \brief Write data package at stream position.
    //! \param[in] data  DataPackage to write.
    //! \returns Either \c true if DataPackage successful written, otherwise \c false.
    //!
    //! \note This method does not change source, index and previous message size in the package header.
    //!       That data is required for serialization but possibly not for your code.
    //----------------------------------------
    bool writePackage(const DataPackage& dataPackage) override;

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
    //! \brief Get the destination Uri.
    //! \return Describing destination Uri of stream.
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
    //! \brief Fix the writer state to append after last package.
    //! \returns Start position where is to append.
    //----------------------------------------
    int64_t fixWriterStateForAppend();

    //========================================
    //! \brief Request resource access with open-mode for file stream.
    //! \param[in] openMode  Stream mode to open the file.
    //----------------------------------------
    void openInternal(const std::ios_base::openmode openMode);

    //========================================
    //! \brief Write IdcDataHeader at current position.
    //! \param[out/in] dh  IdcDataHeader to write.
    //! \return \c True if successful, otherwise \c false.
    //----------------------------------------
    bool writeHeader(IdcDataHeader& dh);

private:
    //========================================
    //!\brief The time stamp of the last header that written.
    //----------------------------------------
    NtpTime m_latestHeaderTimestamp;

    //========================================
    //!\brief The time stamp of the first header that written.
    //----------------------------------------
    NtpTime m_firstHeaderTimestamp;

    //========================================
    //!\brief The size of the last messages that written.
    //----------------------------------------
    uint32_t m_latestMsgSize;

    //========================================
    //!\brief The FrameIndex (default: FrameIndex6130) of the currently open stream.
    //----------------------------------------
    FrameIndexPtr m_frameIndex;

    //========================================
    //!\brief The IdcTrailer (default: IdcTrailer6120) of the currently open stream.
    //----------------------------------------
    IdcTrailerPtr m_idcTrailer;

    //========================================
    //!\brief Stream handle or nullptr if not accessible.
    //----------------------------------------
    IoStreamPtr m_stream;

    //========================================
    //!\brief Stream destination Uri.
    //----------------------------------------
    Uri m_path;

    //========================================
    //!\brief Mutex for ThreadSafe implementation.
    //----------------------------------------
    mutable Mutex m_mutex;
};

//==============================================================================

//========================================
//! \brief Nullable IdcStreamWriter pointer.
//----------------------------------------
using IdcStreamWriterPtr = std::unique_ptr<IdcStreamWriter>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
