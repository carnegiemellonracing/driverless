//==============================================================================
//!\file
//!
//! \brief Translate IcdDataPackage to IdcDataPackage from package stream.
//! GDTP zone occupation to SDK container ZoneOccupationListA000.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 24st, 2025
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>
#include <microvision/common/sdk/io/icd/IcdDataPackage.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>

#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/ZoneOccupationListA000ExporterA000.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate IcdDataPackage to IdcDataPackage from package stream.
//!
//! The translation will be done by datatype name of IcdDataPackage.
//! Zone occupation lists will be translated into ZoneOccupationListA000.
//!
//! \extends DataStreamTranslator<IcdDataPackage, IdcDataPackage>
//------------------------------------------------------------------------------
class IcdToIdcPackageOfZoneOccupationListA000Translator final
  : public common::sdk::DataStreamTranslator<common::sdk::IcdDataPackage, common::sdk::IdcDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type
    //----------------------------------------
    using BaseType = DataStreamTranslator<common::sdk::IcdDataPackage, common::sdk::IdcDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{
        "microvision::plugins::movia::IcdToIdcPackageOfZoneOccupationListA000Translator"};

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
    IcdToIdcPackageOfZoneOccupationListA000Translator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IcdToIdcPackageOfZoneOccupationListA000Translator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IcdToIdcPackageTranslator.
    //----------------------------------------
    IcdToIdcPackageOfZoneOccupationListA000Translator(const IcdToIdcPackageOfZoneOccupationListA000Translator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IcdToIdcPackageOfZoneOccupationListA000Translator(IcdToIdcPackageOfZoneOccupationListA000Translator&&) = delete;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~IcdToIdcPackageOfZoneOccupationListA000Translator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IcdToIdcPackageTranslator.
    //----------------------------------------
    IcdToIdcPackageOfZoneOccupationListA000Translator&
    operator=(const IcdToIdcPackageOfZoneOccupationListA000Translator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IcdToIdcPackageOfZoneOccupationListA000Translator& operator=(IcdToIdcPackageOfZoneOccupationListA000Translator&&)
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

    //========================================
    //! \brief Exporter for zone occupation list.
    //----------------------------------------
    common::sdk::ZoneOccupationListA000ExporterA000 m_exporter;
}; // class IcdToIdcPackageOfZoneOccupationListA000Translator

//==============================================================================
//! \brief Nullable IcdToIdcPackageOfZoneOccupationListA000Translator pointer.
//------------------------------------------------------------------------------
using IcdToIdcPackageOfZoneOccupationListA000TranslatorPtr
    = std::shared_ptr<IcdToIdcPackageOfZoneOccupationListA000Translator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================