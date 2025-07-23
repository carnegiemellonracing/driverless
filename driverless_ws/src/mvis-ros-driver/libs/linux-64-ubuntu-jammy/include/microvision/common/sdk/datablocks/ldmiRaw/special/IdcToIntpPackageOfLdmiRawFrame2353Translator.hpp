//==============================================================================
//! \file
//!
//! \brief Translate idc to intp packages of 0x2353 data containers from data stream.
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

#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2353.hpp>
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
//! \brief Translate idc to intp packages of 0x2353 data containers from data stream.
//! \extends DataStreamTranslator<IdcDataPackage, IntpNetworkDataPackage>
//------------------------------------------------------------------------------
class IdcToIntpPackageOfLdmiRawFrame2353Translator final
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
    static constexpr const char* m_loggerId{"microvision::common::sdk::IdcToIntpPackageOfLdmiRawFrame2353Translator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2353Translator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2353Translator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IdcToIntpPackageOfLdmiRawFrame2353Translator.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2353Translator(const IdcToIntpPackageOfLdmiRawFrame2353Translator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2353Translator(IdcToIntpPackageOfLdmiRawFrame2353Translator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcToIntpPackageOfLdmiRawFrame2353Translator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IdcToIntpPackageOfLdmiRawFrame2353Translator.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2353Translator& operator=(const IdcToIntpPackageOfLdmiRawFrame2353Translator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IdcToIntpPackageOfLdmiRawFrame2353Translator& operator=(IdcToIntpPackageOfLdmiRawFrame2353Translator&&) = delete;

public: // implements DataStreamTranslator<IdcDataPackage, IntpNetworkDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate IdcDataPackage to IntpNetworkDataPackage of LdmiRawFrame2353.
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
    //! \param[in] packageType  INTP package type like A300 etc.
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

}; // class IdcToIntpPackageOfLdmiRawFrame2353Translator

//==============================================================================
//! \brief Nullable IdcToIntpPackageOfLdmiRawFrame2353Translator pointer.
//------------------------------------------------------------------------------
using IdcToIntpPackageOfLdmiRawFrame2353TranslatorPtr = std::shared_ptr<IdcToIntpPackageOfLdmiRawFrame2353Translator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
