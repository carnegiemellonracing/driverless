//==============================================================================
//!\file
//!
//! \brief Translate to idc data packages from package stream.
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

#include <microvision/common/sdk/misc/SharedBufferStream.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate to idc data packages from package stream.
//! \extends DataStreamTranslator<DataPackage, IdcDataPackage>
//------------------------------------------------------------------------------
class DataPackageToIdcTranslator final : public DataStreamTranslator<DataPackage, IdcDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<DataPackage, IdcDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::DataPackageToIdcTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default construct
    //----------------------------------------
    DataPackageToIdcTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    DataPackageToIdcTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of DataPackageToIdcTranslator.
    //----------------------------------------
    DataPackageToIdcTranslator(const DataPackageToIdcTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    DataPackageToIdcTranslator(DataPackageToIdcTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~DataPackageToIdcTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of DataPackageToIdcTranslator.
    //----------------------------------------
    DataPackageToIdcTranslator& operator=(const DataPackageToIdcTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    DataPackageToIdcTranslator& operator=(DataPackageToIdcTranslator&&) = delete;

public: // implements DataStreamTranslator<DataPackage, IdcDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate DataPackage to IdcDataPackage.
    //! \param[in] dataPackage  Input DataPackage to process
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
    //! \brief Unconsumed Data package buffers.
    //----------------------------------------
    ThreadSafe<SharedBufferStream::BufferList> m_buffers;

    //========================================
    //! \brief Post processing function set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;
}; // class DataPackageToIdcTranslator

//==============================================================================
//! \brief Nullable DataPackageToIdcTranslator pointer.
//------------------------------------------------------------------------------
using DataPackageToIdcTranslatorPtr = std::shared_ptr<DataPackageToIdcTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
