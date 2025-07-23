//==============================================================================
//!\file
//!
//! \brief Translate IdcDataPackage to IcdDataPackage from package stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 26, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

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
//! The translation will be done for Scan2341 or Scan2342 which get
//! the ICD measurement point list datatype name and custom data containers
//! which get the name of custom data container (header) as datatype name.
//!
//! \extends DataStreamTranslator<IdcDataPackage, IcdDataPackage>
//------------------------------------------------------------------------------
class IdcToIcdPackageTranslator final : public DataStreamTranslator<IdcDataPackage, IcdDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<IdcDataPackage, IcdDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::IdcToIcdPackageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    IdcToIcdPackageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IdcToIcdPackageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IdcToIcdPackageTranslator.
    //----------------------------------------
    IdcToIcdPackageTranslator(const IdcToIcdPackageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IdcToIcdPackageTranslator(IdcToIcdPackageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcToIcdPackageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IdcToIcdPackageTranslator.
    //----------------------------------------
    IdcToIcdPackageTranslator& operator=(const IdcToIcdPackageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IdcToIcdPackageTranslator& operator=(IdcToIcdPackageTranslator&&) = delete;

public: // implements DataStreamTranslator<IdcDataPackage, DataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate IdcDataPackage to IcdDataPackage.
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
    //! \brief Post processing function set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

}; // class IdcToIcdPackageTranslator

//==============================================================================
//! \brief Nullable IdcToIcdPackageTranslator pointer.
//------------------------------------------------------------------------------
using IdcToIcdPackageTranslatorPtr = std::shared_ptr<IdcToIcdPackageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
