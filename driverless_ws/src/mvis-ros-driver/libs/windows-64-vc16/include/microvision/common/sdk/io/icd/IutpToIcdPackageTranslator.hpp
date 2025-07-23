//==============================================================================
//!\file
//!
//! \brief Translate iutp to icd reading icd header.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 13th, 2022
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/iutp/IutpNetworkDataPackage.hpp>
#include <microvision/common/sdk/io/icd/IcdDataPackage.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate IutpNetworkDataPackage to IcdDataPackage from package stream.
//!
//! The translation interprets the icd header found within the iutp network package
//! payload depending on the datatype and stream id.
//!
//! \extends DataStreamTranslator<IutpNetworkDataPackage, IcdDataPackage>
//------------------------------------------------------------------------------
class IutpToIcdPackageTranslator final : public DataStreamTranslator<IutpNetworkDataPackage, IcdDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<IutpNetworkDataPackage, IcdDataPackage>;

    //========================================
    //! \brief IUTP stream id
    //----------------------------------------
    static constexpr uint8_t iutpStreamId{0x80};

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::IutpToIcdPackageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    IutpToIcdPackageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created icd packages.
    //----------------------------------------
    IutpToIcdPackageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IutpToIcdPackageTranslator.
    //----------------------------------------
    IutpToIcdPackageTranslator(const IutpToIcdPackageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IutpToIcdPackageTranslator(IutpToIcdPackageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IutpToIcdPackageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IutpToIcdPackageTranslator.
    //----------------------------------------
    IutpToIcdPackageTranslator& operator=(const IutpToIcdPackageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IutpToIcdPackageTranslator& operator=(IutpToIcdPackageTranslator&&) = delete;

public: // implements DataStreamTranslator<IdcDataPackage, DataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate IutpNetworkDataPackage to IcdDataPackage.
    //! \param[in] dataPackage  Input IutpNetworkDataPackage to process
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
    //! \brief Post processing function set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

}; // class IutpToIcdPackageTranslator

//==============================================================================
//! \brief Nullable IutpToIcdPackageTranslator pointer.
//------------------------------------------------------------------------------
using IutpToIcdPackageTranslatorPtr = std::shared_ptr<IutpToIcdPackageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
