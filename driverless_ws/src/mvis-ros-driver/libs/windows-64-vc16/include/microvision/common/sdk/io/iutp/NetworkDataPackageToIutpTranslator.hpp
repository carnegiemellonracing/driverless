//==============================================================================
//! \file
//!
//! \brief Translate to iutp data packages from network package stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 26, 2020
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/iutp/IutpNetworkDataPackage.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/logging/logging.hpp>

#include <unordered_map>
#include <chrono>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate to iutp data packages from network package stream.
//! \extends DataStreamTranslator<NetworkDataPackage, IutpNetworkDataPackage>
//------------------------------------------------------------------------------
class NetworkDataPackageToIutpTranslator final : public DataStreamTranslator<NetworkDataPackage, IutpNetworkDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<NetworkDataPackage, IutpNetworkDataPackage>;

    //========================================
    //! \brief Default timeout duration until cleanup.
    //----------------------------------------
    static constexpr uint32_t defaultCleanupTimeoutInMs{1000};

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::NetworkDataPackageToIutpTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

private:
    //========================================
    //! \brief Incomplete fragmented iutp message.
    //----------------------------------------
    struct FragmentedPackage
    {
    public:
        //========================================
        //! \brief Fragmented iutp message.
        //----------------------------------------
        IutpNetworkDataPackagePtr data{};

        //========================================
        //! \brief Count of received fragments for this message.
        //----------------------------------------
        uint16_t fragmentsReceived{0};

        //========================================
        //! \brief Vector has the size of expected fragments
        //!     and if fragment is received the value is true.
        //----------------------------------------
        std::vector<bool> fragments{};

        //========================================
        //! \brief Point in time at which it can be cleanup.
        //----------------------------------------
        std::chrono::high_resolution_clock::time_point cleanupTimestamp{};
    };

    //========================================
    //! \brief Incomplete fragmented iutpmessage store type.
    //----------------------------------------
    using FragmentedPackageMap = std::unordered_map<std::size_t, FragmentedPackage>;

public:
    //========================================
    //! \brief Default construct
    //----------------------------------------
    NetworkDataPackageToIutpTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    NetworkDataPackageToIutpTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IdcTranslator.
    //----------------------------------------
    NetworkDataPackageToIutpTranslator(const NetworkDataPackageToIutpTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    NetworkDataPackageToIutpTranslator(NetworkDataPackageToIutpTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~NetworkDataPackageToIutpTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of NetworkDataPackageToIutpTranslator.
    //----------------------------------------
    NetworkDataPackageToIutpTranslator& operator=(const NetworkDataPackageToIutpTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    NetworkDataPackageToIutpTranslator& operator=(NetworkDataPackageToIutpTranslator&&) = delete;

public: // getter
    //========================================
    //! \brief Get timeout when to cleanup fragment.
    //! \returns Timeout in milliseconds
    //----------------------------------------
    uint32_t getFragmentTimeoutInMs() const;

public: // setter
    //========================================
    //! \brief Set timeout when to cleanup fragment.
    //! \param[in] timeoutInMs  Timeout in milliseconds
    //----------------------------------------
    void setFragmentTimeoutInMs(const uint32_t timeoutInMs);

public: // implements DataStreamTranslator<DataPackage, IutpNetworkDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate NetworkDataPackage to IutpNetworkDataPackage.
    //! \param[in] dataPackage  Input NetworkDataPackage to process
    //! \returns Either \c true if input is valid, otherwise \c false.
    //----------------------------------------
    bool translate(const InputPtr& dataPackage) override;

    //========================================
    //! \brief Clean up the translator state.
    //! \note Call it to get memory free,
    //!       only if no more packages are coming in.
    //----------------------------------------
    void clear() override;

private:
    //========================================
    //! \brief Compute key hash for indexing of incomplete messages.
    //! \param[in] remoteUri        Source endpoint from where the package came.
    //! \param[in] streamId         Id of the data stream.
    //! \param[in] sequenceNumber   Message index of stream.
    //! \returns Combined hash of parameters.
    //----------------------------------------
    std::size_t computeKeyHash(const Uri& remoteUri, const uint8_t streamId, const uint16_t sequenceNo) const;

private:
    //========================================
    //! \brief Timeout when to cleanup fragment.
    //----------------------------------------
    ThreadSafe<uint32_t> m_cleanupTimeoutInMs;

    //========================================
    //! \brief Store of incomplete messages.
    //----------------------------------------
    FragmentedPackageMap m_fragmentedPackages;

    //========================================
    //! \brief Count of incomplete messages dropped because of timeout.
    //----------------------------------------
    uint32_t m_numberOfIncompletePackages;

    //========================================
    //! \brief Count of complete messages sent to the output callback.
    //----------------------------------------
    uint32_t m_numberOfCompletePackages;

    //========================================
    //! \brief Post processing function.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;
}; // class NetworkDataPackageToIutpTranslator

//==============================================================================
//! \brief Nullable NetworkDataPackageToIutpTranslator pointer.
//------------------------------------------------------------------------------
using NetworkDataPackageToIutpTranslatorPtr = std::shared_ptr<NetworkDataPackageToIutpTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
