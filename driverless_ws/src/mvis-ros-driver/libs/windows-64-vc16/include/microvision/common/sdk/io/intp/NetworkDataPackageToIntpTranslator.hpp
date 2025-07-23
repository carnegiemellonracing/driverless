//==============================================================================
//! \file
//!
//! \brief Translate to intp data packages from network package stream.
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

#include <microvision/common/sdk/io/intp/IntpNetworkDataPackage.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>
#include <microvision/common/sdk/misc/crypto/checksum.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate to intp data packages from network package stream.
//! \extends DataStreamTranslator<NetworkDataPackage, IntpNetworkDataPackage>
//------------------------------------------------------------------------------
class NetworkDataPackageToIntpTranslator final : public DataStreamTranslator<NetworkDataPackage, IntpNetworkDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<NetworkDataPackage, IntpNetworkDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::NetworkDataPackageToIntpTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default construct
    //----------------------------------------
    NetworkDataPackageToIntpTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    NetworkDataPackageToIntpTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IdcTranslator.
    //----------------------------------------
    NetworkDataPackageToIntpTranslator(const NetworkDataPackageToIntpTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    NetworkDataPackageToIntpTranslator(NetworkDataPackageToIntpTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~NetworkDataPackageToIntpTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of NetworkDataPackageToIntpTranslator.
    //----------------------------------------
    NetworkDataPackageToIntpTranslator& operator=(const NetworkDataPackageToIntpTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    NetworkDataPackageToIntpTranslator& operator=(NetworkDataPackageToIntpTranslator&&) = delete;

public: // implements DataStreamTranslator<DataPackage, IntpNetworkDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate NetworkDataPackage to IntpNetworkDataPackage.
    //! \param[in] dataPackage  Input NetworkDataPackage to process
    //! \returns Either \c true if input is valid, otherwise \c false.
    //----------------------------------------
    bool translate(const InputPtr& dataPackage) override;

    //========================================
    //! \brief Clean up the translator state.
    //! \note Will nothing do, do not call it.
    //----------------------------------------
    void clear() override;

private:
    //========================================
    //! \brief Post processing function.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

    //========================================
    //! \brief Helper to generate crc32 checksum.
    //----------------------------------------
    crypto::AutosarCrc32 m_checksumGenerator;

}; // class NetworkDataPackageToIntpTranslator

//==============================================================================
//! \brief Nullable NetworkDataPackageToIntpTranslator pointer.
//------------------------------------------------------------------------------
using NetworkDataPackageToIntpTranslatorPtr = std::shared_ptr<NetworkDataPackageToIntpTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
