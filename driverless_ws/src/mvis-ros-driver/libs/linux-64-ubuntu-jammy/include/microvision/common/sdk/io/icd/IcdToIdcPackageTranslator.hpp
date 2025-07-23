//==============================================================================
//!\file
//!
//! \brief Translate IcdDataPackage to IdcDataPackage from package stream.
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

#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/io/IdcDataPackage.hpp>
#include <microvision/common/sdk/io/icd/IcdDataPackage.hpp>
#include <microvision/common/sdk/io/DataStreamTranslator.hpp>
#include <microvision/common/sdk/datablocks/PerceptionDataInfo.hpp>
#include <microvision/common/sdk/datablocks/PerceptionPerformanceInfo.hpp>

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
//! So far will ICD measurement point list translated into Scan2341 or Scan2342
//! depending to there length. And all others will packed into custom data container
//! with computed UUID by datatype name and as well will the datatype name used as name.
//!
//! \extends DataStreamTranslator<IcdDataPackage, IdcDataPackage>
//------------------------------------------------------------------------------
class IcdToIdcPackageTranslator final : public DataStreamTranslator<IcdDataPackage, IdcDataPackage>
{
public:
    //========================================
    //! \brief Definition of base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<IcdDataPackage, IdcDataPackage>;

    //========================================
    //! \brief Function type for lambda function to select setup device id.
    //----------------------------------------
    using DeviceIdSelectorFunction = std::function<uint8_t(const IcdDataPackagePtr&, const PerceptionDataInfo&)>;

public:
    enum IdcHeaderTimestampMode
    {
        Keep                     = 0, //!< keep perception data header timestamp
        ReceiveTime              = 1, //!< use current time
        GdtpMeasurementTimestamp = 2 //!< replace with measurement timestamp from gdtp header
    };

private:
    //========================================
    //! \brief Data type to store/collect scan, scanner info and other data required
    //!        to completely publish a full IDC scan from arriving icd data.
    //----------------------------------------
    struct SensorData
    {
        IcdDataPackagePtr scan{}; //!< Scan data package
        IcdDataPackagePtr scannerInfo{}; //!< Scanner info data package
        IcdDataPackagePtr performanceInfo{}; //!< Performance Info
        PerceptionDataInfo scanDataInfo{}; //!< Scan data info
        PerceptionPerformanceInfo performanceDataInfo{};
    };

    //========================================
    //! \brief Map to index sensor data by id.
    //----------------------------------------
    using SensorDataMapType = std::map<uint64_t, SensorData>;

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::IcdToIdcPackageTranslator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    IcdToIdcPackageTranslator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IcdToIdcPackageTranslator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IcdToIdcPackageTranslator.
    //----------------------------------------
    IcdToIdcPackageTranslator(const IcdToIdcPackageTranslator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IcdToIdcPackageTranslator(IcdToIdcPackageTranslator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IcdToIdcPackageTranslator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IcdToIdcPackageTranslator.
    //----------------------------------------
    IcdToIdcPackageTranslator& operator=(const IcdToIdcPackageTranslator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IcdToIdcPackageTranslator& operator=(IcdToIdcPackageTranslator&&) = delete;

    //========================================
    //! \brief Set device id selector function.
    //! \note If the device id selector is not set the sensor id will downcasted.
    //!       The sensor id will get by reading the PerceptionDataInfo
    //!       at begin of the payload.
    //! \param[in] deviceIdSelector  Device id selector function pointer.
    //----------------------------------------
    void setDeviceIdSelector(const DeviceIdSelectorFunction& deviceIdSelector);

    //========================================
    //! \brief Configure the type of output data containers.
    //! \note Use this to increase throughput performance. If scan or icd output is not
    //! required disabling can gain you a lot of performance.
    //! \param[in] enableScanOutput        Flag to enable Idc scan data container output.
    //! \param[in] enableIcdOutput         Flag to enable icd custom data container passthrough.
    //! \param[in] enableAllCustomOutput   Flag to enable passthrough of all received data as custom data container.
    //! \param[in] usedIdcHeaderTimestamp  Mode for usage of receiving time or gdtp header measurement timestamp for ntp time of idc data header.
    //! \param[in] enableHermesIdcOutput   Flag to enable output of received idc data containers streamed from hermes data bridge.
    //----------------------------------------
    void setOutputConfiguration(const bool enableScanOutput,
                                const bool enableIcdOutput,
                                const bool enableAllCustomOutput,
                                const IdcHeaderTimestampMode usedIdcHeaderTimestamp,
                                const bool enableHermesIdcOutput);

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
    //! \brief Get setup device id.
    //! \param[in] package  Input IdcDataPackage to process
    //! \param[in] header   Perception header of input package.
    //! \returns Setup device id.
    //----------------------------------------
    uint8_t selectDeviceSetupId(const BaseType::InputPtr& package, const PerceptionDataInfo& header);

    //========================================
    //! \brief Publish Scan2341/Scan2342 of collected icd data.
    //! \param[in] data  Collected icd data.
    //! \returns Either \c true if publish successful, otherwise \c false.
    //----------------------------------------
    bool convertAndOutputAsScan(const SensorData& data);

private:
    //========================================
    //! \brief Post processing callback set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

    //========================================
    //! \brief Device id selector function.
    //----------------------------------------
    DeviceIdSelectorFunction m_deviceIdSelector;

    //========================================
    //! \brief Scan conversion of point cloud data enabled.
    //----------------------------------------
    bool m_scanOutputEnabled;

    //========================================
    //! \brief Icd point cloud data passthrough as custom data container enabled.
    //----------------------------------------
    bool m_icdOutputEnabled;

    //========================================
    //! \brief All received data passthrough as custom data container enabled.
    //----------------------------------------
    bool m_allCustomOutputEnabled;

    //========================================
    //! \brief Usage of receiving time for idc data header in custom data container enabled.
    //----------------------------------------
    IdcHeaderTimestampMode m_idcHeaderTimestampMode;

    //========================================
    //! \brief All received idc data containers pass through if enabled.
    //----------------------------------------
    bool m_hermesIdcOutputEnabled;

    //========================================
    //! \brief Previous package size of received packages.
    //----------------------------------------
    ThreadSafe<uint32_t> m_previousPackageSize;

    //========================================
    //! \brief Data per sensor received.
    //----------------------------------------
    SensorDataMapType m_sensorDataMap;

}; // class IcdToIdcPackageTranslator

//==============================================================================
//! \brief Nullable IcdToIdcPackageTranslator pointer.
//------------------------------------------------------------------------------
using IcdToIdcPackageTranslatorPtr = std::shared_ptr<IcdToIdcPackageTranslator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
