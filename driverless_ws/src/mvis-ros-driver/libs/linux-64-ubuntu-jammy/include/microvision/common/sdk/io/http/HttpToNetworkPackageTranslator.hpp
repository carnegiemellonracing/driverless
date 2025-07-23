//==============================================================================
//!\file
//!
//! \brief Definition of translator from HttpNetworkDataPackage to NetworkDataPackage in package stream.
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
//! \brief Translate HttpNetworkDataPackage to NetworkDataPackage from package stream.
//!
//! Whether \a HttpRequestNetworkDataPackage or \a HttpResponseNetworkDataPackage will
//! written in \a NetworkDataPackage payload as ASCII HTTP message.
//!
//! \extends DataStreamTranslator<HttpNetworkDataPackage, NetworkDataPackage>
//------------------------------------------------------------------------------
class HttpToNetworkPackageTranslator final : public DataStreamTranslator<HttpNetworkDataPackage, NetworkDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<HttpNetworkDataPackage, NetworkDataPackage>;

private:
    //========================================
    //! \brief Logger name used by setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::HttpToNetworkPackageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    HttpToNetworkPackageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    HttpToNetworkPackageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of HttpToNetworkPackageTranslator.
    //----------------------------------------
    HttpToNetworkPackageTranslator(const HttpToNetworkPackageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    HttpToNetworkPackageTranslator(HttpToNetworkPackageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~HttpToNetworkPackageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of HttpToNetworkPackageTranslator.
    //----------------------------------------
    HttpToNetworkPackageTranslator& operator=(const HttpToNetworkPackageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    HttpToNetworkPackageTranslator& operator=(HttpToNetworkPackageTranslator&&) = delete;

public: // implements DataStreamTranslator<HttpNetworkDataPackage, NetworkDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate HttpNetworkDataPackage to NetworkDataPackage.
    //! \param[in] dataPackage  Input HttpNetworkDataPackage to process.
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
    //! \brief Verify human readable string.
    //! \param[in] str  String to verify.
    //! \param[in] canBeEmpty  Either \c true if string can be empty, otherwise \c false.
    //! \param[in] canIncludeSpaces  Either \c true if the string is allowed to include blanks, otherwise \c false.
    //! \return Either \c true if string is valid, otherwise \c false.
    //----------------------------------------
    bool verifyString(const std::string& str, const bool canBeEmpty, const bool canIncludeSpaces) const;

private:
    //========================================
    //! \brief Post processing function set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

}; // class HttpToNetworkPackageTranslator

//==============================================================================
//! \brief Nullable HttpToNetworkPackageTranslator pointer.
//------------------------------------------------------------------------------
using HttpToNetworkPackageTranslatorPtr = std::shared_ptr<HttpToNetworkPackageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
