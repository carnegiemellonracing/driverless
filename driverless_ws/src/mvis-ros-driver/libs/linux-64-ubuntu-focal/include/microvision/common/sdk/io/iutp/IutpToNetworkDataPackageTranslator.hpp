//==============================================================================
//! \file
//!
//! \brief Translate to network data packages from iutp package stream.
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

#include <microvision/common/sdk/io/iutp/IutpNetworkDataPackage.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate to network data packages from iutp package stream.
//! \extends DataStreamTranslator<IutpNetworkDataPackage, NetworkDataPackage>
//------------------------------------------------------------------------------
class IutpToNetworkDataPackageTranslator final : public DataStreamTranslator<IutpNetworkDataPackage, NetworkDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<IutpNetworkDataPackage, NetworkDataPackage>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::IutpToNetworkDataPackageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default construct
    //----------------------------------------
    IutpToNetworkDataPackageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IutpToNetworkDataPackageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IdcTranslator.
    //----------------------------------------
    IutpToNetworkDataPackageTranslator(const IutpToNetworkDataPackageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IutpToNetworkDataPackageTranslator(IutpToNetworkDataPackageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IutpToNetworkDataPackageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IutpToNetworkDataPackageTranslator.
    //----------------------------------------
    IutpToNetworkDataPackageTranslator& operator=(const IutpToNetworkDataPackageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IutpToNetworkDataPackageTranslator& operator=(IutpToNetworkDataPackageTranslator&&) = delete;

public: // getter
    //========================================
    //! \brief Get size of fragment in which the message is to split.
    //! \returns Size of fragment.
    //----------------------------------------
    uint16_t getFragmentSize() const;

public: // setter
    //========================================
    //! \brief Set size of fragments in which the message is to split.
    //! \param[in] fragmentSize  Size of fragment.
    //----------------------------------------
    void setFragmentSize(const uint16_t fragmentSize);

public: // implements DataStreamTranslator<IutpNetworkDataPackage, NetworkDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate IutpNetworkDataPackage to NetworkDataPackage.
    //! \param[in] dataPackage  Input IutpNetworkDataPackage to process
    //! \returns Either \c true if input is valid, otherwise \c false.
    //----------------------------------------
    bool translate(const InputPtr& iutpPackage) override;

    //========================================
    //! \brief Clean up the translator state.
    //! \note Will nothing do, do not call it.
    //----------------------------------------
    void clear() override;

private:
    //========================================
    //! \brief Size of the fragment.
    //----------------------------------------
    ThreadSafe<uint16_t> m_fragmentSize;

    //========================================
    //! \brief Post processing function.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;
}; // class DataPackageToIntpTranslator

//==============================================================================
//! \brief Nullable IutpToNetworkDataPackageTranslator pointer.
//------------------------------------------------------------------------------
using IutpToNetworkDataPackageTranslatorPtr = std::shared_ptr<IutpToNetworkDataPackageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
