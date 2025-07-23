//==============================================================================
//! \file
//!
//! \brief Translate idc to intp packages of 0x2352 data containers from data stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 12, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2352.hpp>
#include <microvision/common/sdk/io/intp/IntpNetworkDataPackage.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate idc to intp packages of 0x2352 data containers from data stream.
//! \extends DataStreamTranslator<IdcDataPackage, IntpNetworkDataPackage>
//------------------------------------------------------------------------------
class IdcToIntpPackageOfLdmiRawFrame2352Translator final
  : public DataStreamTranslator<IdcDataPackage, IntpNetworkDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<IdcDataPackage, IntpNetworkDataPackage>;

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
    //! \brief Default constructor.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2352Translator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2352Translator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IdcToIntpPackageOfLdmiRawFrame2352Translator.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2352Translator(const IdcToIntpPackageOfLdmiRawFrame2352Translator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2352Translator(IdcToIntpPackageOfLdmiRawFrame2352Translator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcToIntpPackageOfLdmiRawFrame2352Translator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IdcToIntpPackageOfLdmiRawFrame2352Translator.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2352Translator& operator=(const IdcToIntpPackageOfLdmiRawFrame2352Translator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2352Translator& operator=(IdcToIntpPackageOfLdmiRawFrame2352Translator&&) = delete;

public: // implements DataStreamTranslator<IdcDataPackage, IntpNetworkDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate IdcDataPackage to IntpNetworkDataPackage of LdmiRawFrame2352.
    //! \param[in] dataPackage  Input IdcDataPackage to process
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
    //! \brief Create and publish (via callback) intp package.
    //! \param[in] dataPackage  Current IdcDataPackage to translate.
    //! \param[in] packageType  INTP package type like A200 etc.
    //! \param[in] payload      Partial payload for INTP package.
    //----------------------------------------
    void createAndPublishIntpPackage(const InputPtr& dataPackage,
                                     const IntpPackageType packageType,
                                     const SharedBuffer& payload);

private:
    //========================================
    //! \brief Post processing function set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

}; // class IdcToIntpPackageOfLdmiRawFrame2352Translator

//==============================================================================
//! \brief Nullable IdcToIntpPackageOfLdmiRawFrame2352Translator pointer.
//------------------------------------------------------------------------------
using IdcToIntpPackageOfLdmiRawFrame2352TranslatorPtr = std::shared_ptr<IdcToIntpPackageOfLdmiRawFrame2352Translator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
