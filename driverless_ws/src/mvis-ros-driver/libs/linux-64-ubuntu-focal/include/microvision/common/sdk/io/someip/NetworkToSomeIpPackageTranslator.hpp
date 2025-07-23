//==============================================================================
//! \file
//!
//! \brief Translate to SOME/IP data packages from tcp package stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 29, 2023
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/DataStreamTranslator.hpp>
#include <microvision/common/sdk/io/NetworkDataPackage.hpp>
#include <microvision/common/sdk/io/someip/SomeIpDataPackage.hpp>

#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/misc/SharedBufferStream.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate to SomeIpDataPackage from tcp package stream.
//! \extends DataStreamTranslator<NetworkDataPackage, SomeIpDataPackage>
//------------------------------------------------------------------------------
class NetworkToSomeIpPackageTranslator final : public DataStreamTranslator<NetworkDataPackage, SomeIpDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<NetworkDataPackage, SomeIpDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::NetworkToSomeIpPackageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default construct
    //----------------------------------------
    NetworkToSomeIpPackageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created SOME/IP data packages.
    //----------------------------------------
    NetworkToSomeIpPackageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of NetworkToSomeIpPackageTranslator.
    //----------------------------------------
    NetworkToSomeIpPackageTranslator(const NetworkToSomeIpPackageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    NetworkToSomeIpPackageTranslator(NetworkToSomeIpPackageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~NetworkToSomeIpPackageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of NetworkToSomeIpPackageTranslator.
    //----------------------------------------
    NetworkToSomeIpPackageTranslator& operator=(const NetworkToSomeIpPackageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    NetworkToSomeIpPackageTranslator& operator=(NetworkToSomeIpPackageTranslator&&) = delete;

public:
    //========================================
    //! \brief Add service channel configuration.
    //! \param[in] channelConfig  Channel configuration.
    //----------------------------------------
    void addChannelConfig(const SomeIpDataPackage::ChannelConfig& channelConfig);

public: // implements DataStreamTranslator<NetworkDataPackage, SomeIpDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback called when translation is done.
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate NetworkDataPackage to SOME/IP data packages.
    //! \param[in] dataPackage  Input NetworkDataPackage to process.
    //! \returns Either \c true if input is valid, otherwise \c false.
    //----------------------------------------
    bool translate(const InputPtr& dataPackage) override;

    //========================================
    //! \brief Clean up the translator state.
    //! \note Call it to free memory,
    //!       only if no more packages are coming in.
    //----------------------------------------
    void clear() override;

private:
    //========================================
    //! \brief Reset translate status.
    //----------------------------------------
    void reset();

    //========================================
    //! \brief Parse SOME/IP package and publish when successful.
    //! \param[in] currentPackage   SOME/IP current processed package.
    //! \param[in] configuration    SOME/IP channel configuration.
    //! \param[in] messagePayload   SOME/IP message payload.
    //! \return Either \c true if successful, otherwise \c false.
    //----------------------------------------
    bool handleSomeipMessage(const NetworkDataPackagePtr& currentPackage,
                             const SomeIpDataPackage::ChannelConfig& configuration,
                             const SharedBuffer& messagePayload);

private:
    //========================================
    //! \brief Post processing function set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

    //========================================
    //! \brief Service channel configurations.
    //----------------------------------------
    std::vector<SomeIpDataPackage::ChannelConfig> m_channels;

    //========================================
    //! \brief Number of SOME/IP packages translated.
    //----------------------------------------
    std::size_t m_numberOfSomeIpPackages;

    //========================================
    //! \brief Unconsumed Data package buffers.
    //----------------------------------------
    SharedBufferStream::BufferList m_buffers;

    //========================================
    //! \brief E2E Profile7 checksum calculator.
    //----------------------------------------
    SomeIpDataPackage::E2EProfile7CrcCalculatorType m_checksumCalculator;
}; // class NetworkToSomeIpPackageTranslator

//==============================================================================
//! \brief Nullable NetworkToSomeIpPackageTranslator pointer.
//------------------------------------------------------------------------------
using NetworkToSomeIpPackageTranslatorPtr = std::shared_ptr<NetworkToSomeIpPackageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
