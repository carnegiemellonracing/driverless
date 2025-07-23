//==============================================================================
//!\file
//!
//! \brief Translate Image2404 IdcDataPackage to IcdDataPackage with GDTP image from package stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 7th, 2025
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>
#include <microvision/common/sdk/io/icd/IcdDataPackage.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate IdcDataPackage to IcdDataPackage from package stream.
//!
//! The translation will be done by datatype name of IcdDataPackage.
//! Image2404 will be translated into icd images.
//!
//! \extends DataStreamTranslator<IdcDataPackage, IcdDataPackage>
//------------------------------------------------------------------------------
class IdcToIcdPackageOfImage2404Translator final
  : public common::sdk::DataStreamTranslator<common::sdk::IdcDataPackage, common::sdk::IcdDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type
    //----------------------------------------
    using BaseType = DataStreamTranslator<common::sdk::IdcDataPackage, common::sdk::IcdDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::plugins::movia::IdcToIcdPackageOfImage2404Translator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

    //========================================
    //! \brief Image topic name.
    //----------------------------------------
    static const std::string imageTopicName;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    IdcToIcdPackageOfImage2404Translator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IdcToIcdPackageOfImage2404Translator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IcdToIdcPackageTranslator.
    //----------------------------------------
    IdcToIcdPackageOfImage2404Translator(const IdcToIcdPackageOfImage2404Translator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IdcToIcdPackageOfImage2404Translator(IdcToIcdPackageOfImage2404Translator&&) = delete;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~IdcToIcdPackageOfImage2404Translator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IcdToIdcPackageTranslator.
    //----------------------------------------
    IdcToIcdPackageOfImage2404Translator& operator=(const IdcToIcdPackageOfImage2404Translator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IdcToIcdPackageOfImage2404Translator& operator=(IdcToIcdPackageOfImage2404Translator&&) = delete;

public: // implements DataStreamTranslator<DataPackage, IdcDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate IcdDataPackage to IdcDataPackage.
    //! \param[in] dataPackage  Input IcdDataPackage to process
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

    //========================================
    //! \brief Previous package size of received packages.
    //----------------------------------------
    common::sdk::ThreadSafe<uint32_t> m_previousPackageSize;
}; // class IdcToIcdPackageOfImage2404Translator

//==============================================================================
//! \brief Nullable IdcToIcdPackageOfImage2404Translator pointer.
//------------------------------------------------------------------------------
using IdcToIcdPackageOfImage2404TranslatorPtr = std::shared_ptr<IdcToIcdPackageOfImage2404Translator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================