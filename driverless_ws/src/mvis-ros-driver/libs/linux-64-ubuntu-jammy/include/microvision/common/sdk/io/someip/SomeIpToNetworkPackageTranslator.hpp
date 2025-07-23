//==============================================================================
//! \file
//!
//! \brief Translate to network data package from SOME/IP package.
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
//! \brief Translate to network data package from SOME/IP package.
//! \extends DataStreamTranslator<SomeIpDataPackage, NetworkDataPackage>
//------------------------------------------------------------------------------
class SomeIpToNetworkPackageTranslator final : public DataStreamTranslator<SomeIpDataPackage, NetworkDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<SomeIpDataPackage, NetworkDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::SomeIpToNetworkPackageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default construct
    //----------------------------------------
    SomeIpToNetworkPackageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created network data packages.
    //----------------------------------------
    SomeIpToNetworkPackageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of SomeIpToNetworkPackageTranslator.
    //----------------------------------------
    SomeIpToNetworkPackageTranslator(const SomeIpToNetworkPackageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    SomeIpToNetworkPackageTranslator(SomeIpToNetworkPackageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~SomeIpToNetworkPackageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of SomeIpToNetworkPackageTranslator.
    //----------------------------------------
    SomeIpToNetworkPackageTranslator& operator=(const SomeIpToNetworkPackageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    SomeIpToNetworkPackageTranslator& operator=(SomeIpToNetworkPackageTranslator&&) = delete;

public:
    //========================================
    //! \brief Add service channel configuration.
    //! \param[in] channelConfig  Channel configuration.
    //----------------------------------------
    void addChannelConfig(const SomeIpDataPackage::ChannelConfig& channelConfig);

    //========================================
    //! \brief Set flag to enable auto flash.
    //! \param[in] autoFlush  Either \c true if every package should be flushed, otherwise \c false.
    //----------------------------------------
    void setAutoFlush(const bool autoFlush);

    //========================================
    //! \brief Flush current message whether it exists.
    //----------------------------------------
    void flush();

public: // implements DataStreamTranslator<SomeIpDataPackage, NetworkDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback called when translation is done.
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate SomeIpDataPackage to network data packages.
    //! \param[in] dataPackage  Input SomeIpDataPackage to process.
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
    //! \brief E2E Profile7 counter.
    //----------------------------------------
    uint32_t m_e2eCounter;

    //========================================
    //! \brief E2E Profile7 checksum calculator.
    //----------------------------------------
    SomeIpDataPackage::E2EProfile7CrcCalculatorType m_checksumCalculator;

    //========================================
    //! \brief Enable auto flush for every package.
    //----------------------------------------
    bool m_autoFlush;

    //========================================
    //! \brief Current NetworkDataPackage to flush.
    //----------------------------------------
    NetworkDataPackagePtr m_currentPackageToFlush;
}; // class SomeIpToNetworkPackageTranslator

//==============================================================================
//! \brief Nullable SomeIpToNetworkPackageTranslator pointer.
//------------------------------------------------------------------------------
using SomeIpToNetworkPackageTranslatorPtr = std::shared_ptr<SomeIpToNetworkPackageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
