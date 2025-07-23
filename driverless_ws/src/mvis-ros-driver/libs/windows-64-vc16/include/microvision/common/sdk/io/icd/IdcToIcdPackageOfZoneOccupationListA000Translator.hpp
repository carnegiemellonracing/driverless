//==============================================================================
//!\file
//!
//! \brief Translate ZoneOccupationListA000 IdcDataPackage to IcdDataPackage with GDTP zone occupation list from package stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 24th, 2025
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
//! ZoneOccupationListA000 will be translated into icd zone occupation lists.
//!
//! \extends DataStreamTranslator<IdcDataPackage, IcdDataPackage>
//------------------------------------------------------------------------------
class IdcToIcdPackageOfZoneOccupationListA000Translator final
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
    static constexpr const char* m_loggerId{
        "microvision::plugins::movia::IdcToIcdPackageOfZoneOccupationListA000Translator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

    //========================================
    //! \brief Zone occupation list topic name.
    //----------------------------------------
    static const std::string icdZoneOccupationListTopicName;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    IdcToIcdPackageOfZoneOccupationListA000Translator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IdcToIcdPackageOfZoneOccupationListA000Translator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IcdToIdcPackageTranslator.
    //----------------------------------------
    IdcToIcdPackageOfZoneOccupationListA000Translator(const IdcToIcdPackageOfZoneOccupationListA000Translator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IdcToIcdPackageOfZoneOccupationListA000Translator(IdcToIcdPackageOfZoneOccupationListA000Translator&&) = delete;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~IdcToIcdPackageOfZoneOccupationListA000Translator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IcdToIdcPackageTranslator.
    //----------------------------------------
    IdcToIcdPackageOfZoneOccupationListA000Translator&
    operator=(const IdcToIcdPackageOfZoneOccupationListA000Translator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IdcToIcdPackageOfZoneOccupationListA000Translator& operator=(IdcToIcdPackageOfZoneOccupationListA000Translator&&)
        = delete;

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
}; // class IdcToIcdPackageOfZoneOccupationListA000Translator

//==============================================================================
//! \brief Nullable IdcToIcdPackageOfZoneOccupationListA000Translator pointer.
//------------------------------------------------------------------------------
using IdcToIcdPackageOfZoneOccupationListA000TranslatorPtr
    = std::shared_ptr<IdcToIcdPackageOfZoneOccupationListA000Translator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================