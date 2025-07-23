//==============================================================================
//!\file
//!
//! \brief Definition of translator from NetworkDataPackage to HttpNetworkDataPackage used for a package stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 06, 2022
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/http/HttpNetworkDataPackage.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate NetworkDataPackage to HttpNetworkDataPackage from package stream.
//!
//! As input it expect an ASCII network message to parse the HTTP header parameters
//! into \a HttpRequestNetworkDataPackage or \a HttpResponseNetworkDataPackage.
//!
//! \note The HTTP message body will stored as payload on \a HttpNetworkDataPackage.
//! \extends DataStreamTranslator<NetworkDataPackage, HttpNetworkDataPackage>
//------------------------------------------------------------------------------
class NetworkToHttpPackageTranslator final : public DataStreamTranslator<NetworkDataPackage, HttpNetworkDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<NetworkDataPackage, HttpNetworkDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::NetworkToHttpPackageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    NetworkToHttpPackageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    NetworkToHttpPackageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of NetworkToHttpPackageTranslator.
    //----------------------------------------
    NetworkToHttpPackageTranslator(const NetworkToHttpPackageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    NetworkToHttpPackageTranslator(NetworkToHttpPackageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~NetworkToHttpPackageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of NetworkToHttpPackageTranslator.
    //----------------------------------------
    NetworkToHttpPackageTranslator& operator=(const NetworkToHttpPackageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    NetworkToHttpPackageTranslator& operator=(NetworkToHttpPackageTranslator&&) = delete;

public: // implements DataStreamTranslator<NetworkDataPackage, HttpNetworkDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate NetworkDataPackage to HttpNetworkDataPackage.
    //! \param[in] dataPackage  Input NetworkDataPackage to process.
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
    //! \brief Post processing callback set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

}; // class NetworkToHttpPackageTranslator

//==============================================================================
//! \brief Nullable NetworkToHttpPackageTranslator pointer.
//------------------------------------------------------------------------------
using NetworkToHttpPackageTranslatorPtr = std::shared_ptr<NetworkToHttpPackageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
