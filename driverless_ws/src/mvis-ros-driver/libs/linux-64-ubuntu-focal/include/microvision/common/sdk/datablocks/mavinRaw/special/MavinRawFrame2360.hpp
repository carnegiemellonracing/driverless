//==============================================================================
//! \file
//!
//! \brief Data type to store reassembled mavin raw frame coming from MicroVision MAVIN LiDAR sensor.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 27th, 2023
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SharedBuffer.hpp>
#include <microvision/common/sdk/misc/Optional.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/mavinRaw/special/MavinRawSensorInfoIn2360.hpp>

#include <boost/property_tree/ptree.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data type to store reassembled mavin raw frame.
//------------------------------------------------------------------------------
class MavinRawFrame2360 final : public SpecializedDataContainer
{
public:
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;

    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    static constexpr const char* containerType{"sdk.specialcontainer.MavinRawFrame2360"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::MavinRawFrame2360";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    MavinRawFrame2360();

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    MavinRawFrame2360(const MavinRawFrame2360& other);

    //========================================
    //! \brief Move constructor.
    //----------------------------------------
    MavinRawFrame2360(MavinRawFrame2360&& other);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~MavinRawFrame2360() override;

public:
    //========================================
    //! \brief Move assignment operator.
    //! \param[in] other  Another instance of raw frame.
    //! \return Reference of this.
    //----------------------------------------
    MavinRawFrame2360& operator=(MavinRawFrame2360&& other);

    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Another instance of raw frame.
    //! \return Reference of this.
    //----------------------------------------
    MavinRawFrame2360& operator=(const MavinRawFrame2360& other);

public:
    //========================================
    //! \brief Compares two ldmi raw frames for equality.
    //! \param[in] lhs  Ldmi raw frame 2354.
    //! \param[in] rhs  Ldmi raw frame 2354.
    //! \returns Either \c true if both frames are equal or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const MavinRawFrame2360& lhs, const MavinRawFrame2360& rhs);

    //========================================
    //! \brief Compares two ldmi raw frames for inequality.
    //! \param[in] lhs  Ldmi raw frame 2354.
    //! \param[in] rhs  Ldmi raw frame 2354.
    //! \returns Either \c true if both frames are unequal or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const MavinRawFrame2360& lhs, const MavinRawFrame2360& rhs);

public: // DataContainerBase implementation
    uint64_t getClassIdHash() const override;

public: // getter
    //========================================
    //! \brief Get format info extracted from header.
    //! \returns Format info extracted from header.
    //----------------------------------------
    const MavinRawSensorInfoIn2360& getSensorInfo() const;

    //========================================
    //! \brief Get header data of frame.
    //! \returns Header data of frame.
    //----------------------------------------
    const SharedBuffer& getHeaderData() const;

    //========================================
    //! \brief Get header as boost property tree.
    //! \returns Header as boost property tree.
    //----------------------------------------
    const boost::property_tree::ptree& getHeaderAsBoostPropertyTree() const;

    //========================================
    //! \brief Get frame data format extracted from header.
    //! \returns Frame data format.
    //----------------------------------------
    const MavinDataFormat& getFrameDataFormatFromHeader() const;

    //========================================
    //! \brief Get frame data format version extracted from header.
    //! \returns Frame data format version.
    //----------------------------------------
    const MavinDataFormatVersion& getFrameDataFormatVersionFromHeader() const;

    //========================================
    //! \brief Get point cloud data of frame.
    //! \returns Point cloud data of frame.
    //----------------------------------------
    const SharedBuffer& getPointCloudData() const;

    //========================================
    //! \brief Get complete frame data.
    //! \returns Frame data.
    //----------------------------------------
    const SharedBuffer& getFrameData() const;

public: // setter
    //========================================
    //! \brief Set sensor info.
    //! \param[in] sensorInfo  Sensor info.
    //----------------------------------------
    void setSensorInfo(const MavinRawSensorInfoIn2360& sensorInfo);

    //========================================
    //! \brief Set complete frame data.
    //! \param[in] frameData  Frame data.
    //----------------------------------------
    bool setFrameData(const SharedBuffer& frameData);

private:
    //========================================
    //! \brief Read JSON header from \a frameData.
    //! \param[in] frameData  Frame data.
    //! \return Either \c true if successful read, otherwise \c false.
    //----------------------------------------
    bool readHeader(const SharedBuffer& frameData);

    //========================================
    //! \brief Select pointcloud data from \a frameData.
    //! \param[in] frameData  Frame data.
    //! \return Either \c true if successful read, otherwise \c false.
    //----------------------------------------
    bool preparePointCloudData(const SharedBuffer& frameData);

    //========================================
    //! \brief Reset frame data.
    //----------------------------------------
    void reset();

private:
    //========================================
    //! \brief Format info extracted from header.
    //----------------------------------------
    MavinRawSensorInfoIn2360 m_sensorInfo;

    //========================================
    //! \brief Header data
    //----------------------------------------
    SharedBuffer m_headerData;

    //========================================
    //! \brief Header as boost property tree
    //----------------------------------------
    boost::property_tree::ptree m_headerPropertyTree;

    //========================================
    //! \brief Frame data format extracted from header.
    //----------------------------------------
    MavinDataFormat m_frameDataFormat;

    //========================================
    //! \brief Frame data format version extracted from header.
    //----------------------------------------
    MavinDataFormatVersion m_frameDataFormatVersion;

    //========================================
    //! \brief Point cloud data
    //----------------------------------------
    SharedBuffer m_pointCloudData;

    //========================================
    //! \brief Frame data.
    //----------------------------------------
    SharedBuffer m_frameData;
};

//==============================================================================
//! \brief Nullable MavinRawFrame2360 pointer.
//------------------------------------------------------------------------------
using MavinRawFrame2360Ptr = std::shared_ptr<MavinRawFrame2360>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
