//==============================================================================
//! \file
//!
//! \brief Translate intp to idc packages of 0x2352 data containers from data stream.
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
#include <microvision/common/sdk/misc/ThreadSafe.hpp>
#include <microvision/common/sdk/misc/Uri.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Translate intp to idc packages of 0x2352 data containers from data stream.
//! \extends DataStreamTranslator<IntpNetworkDataPackage, IdcDataPackage>
//------------------------------------------------------------------------------
class IntpToIdcPackageOfLdmiRawFrame2352Translator final
  : public DataStreamTranslator<IntpNetworkDataPackage, IdcDataPackage>
{
public:
    //========================================
    //! \brief Definition of the base type.
    //----------------------------------------
    using BaseType = DataStreamTranslator<IntpNetworkDataPackage, IdcDataPackage>;

    //========================================
    //! \brief Default timeout duration until cleanup.
    //----------------------------------------
    static constexpr uint32_t defaultCleanupTimeoutInMs{1000};

private:
    //========================================
    //! \brief Timeout duration clock type.
    //----------------------------------------
    using CleanupClockType = std::chrono::high_resolution_clock;

    //========================================
    //! \brief Frame data received so far.
    //----------------------------------------
    struct FrameData
    {
    public:
        //========================================
        //! \brief Frame data pointer.
        //----------------------------------------
        LdmiRawFrame2352Ptr frame{};

        //========================================
        //! \brief Point in time at which it can be cleaned up.
        //----------------------------------------
        CleanupClockType::time_point cleanupTimestamp{};
    };

    //========================================
    //! \brief Sensor data received so far.
    //----------------------------------------
    struct SensorData
    {
    public: // types used
        //========================================
        //! \brief Map to index incomplete frames by id which is unique for setups.
        //----------------------------------------
        using FrameIndexerType = std::map<uint32_t, FrameData>;

    public: // data
        //========================================
        //! \brief Sensor number by which it is indexed.
        //----------------------------------------
        uint8_t sensorNumber{};

        //========================================
        //! \brief Previous published frame size for IdcDataHeader.
        //----------------------------------------
        uint32_t previousPublishedFrameSize{0};

        //========================================
        //! \brief Sensor data received by udp destination.
        //----------------------------------------
        Uri source{};

        //========================================
        //! \brief Ldmi frame indexed by frame id.
        //----------------------------------------
        FrameIndexerType frames{};
    };

    //========================================
    //! \brief Map to index sensor data by id.
    //----------------------------------------
    using SensorDataMapType = std::map<uint8_t, SensorData>;

    //========================================
    //! \brief Map to index static frame config by id.
    //----------------------------------------
    using StaticInfoIndexerType = std::map<uint64_t, LdmiRawStaticInfoIn2352>;

private:
    //========================================
    //! \brief Logger name to set up configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::IntpToIdcPackageOfLdmiRawFrame2352Translator"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    IntpToIdcPackageOfLdmiRawFrame2352Translator();

    //========================================
    //! \brief Construct instance with output callback.
    //! \param[in] outputCallback  Output callback for created idc packages.
    //----------------------------------------
    IntpToIdcPackageOfLdmiRawFrame2352Translator(const BaseType::OutputCallback& outputCallback);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Other instance of IntpToIdcPackageOfLdmiRawFrame2352Translator.
    //----------------------------------------
    IntpToIdcPackageOfLdmiRawFrame2352Translator(const IntpToIdcPackageOfLdmiRawFrame2352Translator& other);

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    IntpToIdcPackageOfLdmiRawFrame2352Translator(IntpToIdcPackageOfLdmiRawFrame2352Translator&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IntpToIdcPackageOfLdmiRawFrame2352Translator() override;

public:
    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Other instance of IntpToIdcPackageOfLdmiRawFrame2352Translator.
    //----------------------------------------
    IntpToIdcPackageOfLdmiRawFrame2352Translator& operator=(const IntpToIdcPackageOfLdmiRawFrame2352Translator& other);

    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    IntpToIdcPackageOfLdmiRawFrame2352Translator& operator=(IntpToIdcPackageOfLdmiRawFrame2352Translator&&) = delete;

public: // getter
    //========================================
    //! \brief Get timeout when to clean up fragment.
    //! \returns Timeout in milliseconds
    //----------------------------------------
    uint32_t getFrameTimeoutInMs() const;

public: // setter
    //========================================
    //! \brief Set timeout when to clean up fragment.
    //! \param[in] timeoutInMs  Timeout in milliseconds
    //----------------------------------------
    void setFrameTimeoutInMs(const uint32_t timeoutInMs);

    //========================================
    //! \brief Set static info to reassemble ldmi raw frames.
    //! \param[in] staticInfo  Static info of ldmi raw frame.
    //----------------------------------------
    void setStaticInfo(const LdmiRawStaticInfoIn2352& staticInfo);

public: // implements DataStreamTranslator<IntpNetworkDataPackage, IdcDataPackage>
    //========================================
    //! \brief Set output callback which is called when output is complete.
    //! \param[in] callback  Output callback
    //----------------------------------------
    void setOutputCallback(const OutputCallback& callback) override;

    //========================================
    //! \brief Translate IntpNetworkDataPackage to IdcDataPackage of LdmiRawFrame2352.
    //! \param[in] intpPackage  Input IntpNetworkDataPackage to process
    //! \returns Either \c true if input is valid, otherwise \c false.
    //----------------------------------------
    bool translate(const InputPtr& intpPackage) override;

    //========================================
    //! \brief Clean up the translator state.
    //! \note Call it to get memory free,
    //!       only if no more packages are coming in.
    //----------------------------------------
    void clear() override;

private:
    //========================================
    //! \brief Store ldmi static info on sensor data map.
    //!
    //! Complete ldmi packages (config, header, row[1,n], footer) will be published.
    //!
    //! \param[in, out] data        Sensor data store.
    //! \param[in]      staticInfo  Ldmi static info.
    //!
    //! \return Either \c true if the processing was successful, otherwise \c false.
    //----------------------------------------
    bool storeLdmiStaticInfo(SensorData& data, const LdmiRawStaticInfoIn2352& staticInfo);

    //========================================
    //! \brief Store ldmi header on sensor data map.
    //!
    //! Complete ldmi packages (config, header, row[1,n], footer) will be published.
    //!
    //! \param[in, out] data         Sensor data store.
    //! \param[in]      frameHeader  Frame header.
    //!
    //! \return Either \c true if the processing was successful, otherwise \c false.
    //----------------------------------------
    bool storeLdmiHeader(SensorData& data, const LdmiRawFrameHeaderIn2352& frameHeader);

    //========================================
    //! \brief Store ldmi row on sensor data map.
    //!
    //! Complete ldmi packages (config, header, row[1,n], footer) will be published.
    //!
    //! \param[in, out] data      Sensor data store.
    //! \param[in]      frameRow  Frame row.
    //!
    //! \return Either \c true if the processing was successful, otherwise \c false.
    //----------------------------------------
    bool storeLdmiRow(SensorData& data, const LdmiRawFrameRowIn2352& frameRow);

    //========================================
    //! \brief Store ldmi footer on sensor data map.
    //!
    //! Complete ldmi packages (config, header, row[1,n], footer) will be published.
    //!
    //! \param[in, out] data         Sensor data store.
    //! \param[in]      frameFooter  Frame footer.
    //!
    //! \return Either \c true if the processing was successful, otherwise \c false.
    //----------------------------------------
    bool storeLdmiFooter(SensorData& data, const LdmiRawFrameFooterIn2352& frameFooter);

    //========================================
    //! \brief Get or add a frame from sensor data.
    //!
    //! Update cleaning timeout and set static info if not done yet.
    //!
    //! \param[in, out] data     Sensor data store.
    //! \param[in]      frameId  Frame id.
    //!
    //! \return Iterator which points to new or existing frame of sensor.
    //----------------------------------------
    SensorData::FrameIndexerType::iterator initFrame(SensorData& data, const uint32_t frameId);

    //========================================
    //! \brief Publish complete ldmi package(config, header, row[1,n], footer) to listeners.
    //!
    //! Data packages depending on the listener/streamer will be built.
    //!
    //! \param[in] sensorNumber      Sensor number.
    //! \param[in, out] ldmiFrameIt  Complete ldmi package.
    //!
    //! \return Either \c true if the processing was successful, otherwise \c false.
    //----------------------------------------
    bool publishLdmiFrame(SensorData& data, SensorData::FrameIndexerType::iterator& ldmiFrameIt);

    //========================================
    //! \brief Log a detailed warning about an incomplete frame.
    //!
    //! After timeout or when clear is called incomplete frames show a warning indicating why incomplete.
    //!
    //! \param[in] incomplete  Incomplete data stored.
    //! \param[in] timeout     Timeout value when this happened. \c 0 for clear/no timeout.
    //----------------------------------------
    void logIncompleteFrameWarning(const FrameData& incomplete, uint32_t timeout = 0U) const;

private:
    //========================================
    //! \brief Post processing function set by translator user.
    //----------------------------------------
    BaseType::OutputCallback m_outputCallback;

    //========================================
    //! \brief Data per sensor received.
    //----------------------------------------
    SensorDataMapType m_sensorDataMap;

    //========================================
    //! \brief Ldmi static info indexed by reference id.
    //----------------------------------------
    StaticInfoIndexerType m_staticInfos;

    //========================================
    //! \brief Timeout duration until cleanup.
    //----------------------------------------
    ThreadSafe<uint32_t> m_cleanupTimeoutInMs;

    //========================================
    //! \brief Timestamp when to cleanup fragments next time.
    //----------------------------------------
    CleanupClockType::time_point m_nextCleanup;

}; // class IntpToIdcPackageOfLdmiRawFrame2352Translator

//==============================================================================
//! \brief Nullable IntpToIdcPackageOfLdmiRawFrame2352Translator pointer.
//------------------------------------------------------------------------------
using IntpToIdcPackageOfLdmiRawFrame2352TranslatorPtr = std::shared_ptr<IntpToIdcPackageOfLdmiRawFrame2352Translator>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
