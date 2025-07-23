//==============================================================================
//! \file
//!
//! \brief Translate to network data packages from intp package stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Dec 1, 2020
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
//! \brief Translate to network data packages from intp package stream.
//! \extends DataStreamTranslator<IntpNetworkDataPackage, NetworkDataPackage>
//------------------------------------------------------------------------------
class IntpToNetworkDataPackageTranslator final : public DataStreamTranslator<IntpNetworkDataPackage, NetworkDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<IntpNetworkDataPackage, NetworkDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::IntpToNetworkDataPackageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    IntpToNetworkDataPackageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IntpToNetworkDataPackageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IdcTranslator.
    //----------------------------------------
    IntpToNetworkDataPackageTranslator(const IntpToNetworkDataPackageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IntpToNetworkDataPackageTranslator(IntpToNetworkDataPackageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IntpToNetworkDataPackageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IntpToNetworkDataPackageTranslator.
    //----------------------------------------
    IntpToNetworkDataPackageTranslator& operator=(const IntpToNetworkDataPackageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IntpToNetworkDataPackageTranslator& operator=(IntpToNetworkDataPackageTranslator&&) = delete;

public: // implements DataStreamTranslator<IntpNetworkDataPackage, NetworkDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate IntpNetworkDataPackage to NetworkDataPackage.
    //! \param[in] dataPackage  Input IntpNetworkDataPackage to process
    //! \returns Either \c true if input is valid, otherwise \c false.
    //----------------------------------------
    bool translate(const InputPtr& intpPackage) override;

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

}; // class IntpToNetworkDataPackageTranslator

//==============================================================================
//! \brief Nullable IntpToNetworkDataPackageTranslator pointer.
//------------------------------------------------------------------------------
using IntpToNetworkDataPackageTranslatorPtr = std::shared_ptr<IntpToNetworkDataPackageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
