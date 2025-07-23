//==============================================================================
//! \file movia.cpp
//!
//! \brief This module contains a ROS2 node to integrate a MOVIA Solid State
//!        Flash LiDAR sensor into an existing ROS2 environment.
//!
//! Adapt CMakeLists.txt target and path entries to your system. Set MVIS_SDK_PLUGINS_INSTALL_PATH.
//! Build with: colcon build
//! Prepare with: source ./install/setup.sh
//! Run with: ros2 run movia movia --ros-args --params-file ./config/default.yaml
//!
//! To debug build with: colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCXX_STANDARD=14
//! To debug run with: ros2 run --prefix 'gdbserver localhost:3000' movia movia --ros-args --params-file ./config/default.yaml
//! Connect to gdb with remote debug: localhost:3000
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//==============================================================================

// ROS2 specific includes
#include <rclcpp/rclcpp.hpp>
#include <rcpputils/asserts.hpp>
// Include ROS2 messages
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

// MICROVISION specific includes
#include <microvision/common/sdk/idc.hpp>
#include <microvision/common/plugin_framework/plugin_loader/PluginLoader.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2404.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340.hpp>
#include <microvision/common/sdk/devices/MicroVisionMoviaSensorDevice.hpp>
#include <microvision/common/sdk/extension/DeviceFactory.hpp>
#include <microvision/common/sdk/extension/StreamReaderFactory.hpp>
#include <microvision/common/sdk/jpegsupport/jmemio.h>

// Include basic system libraries
#include <chrono>
#include <memory>
#include <string>

//========================================
// MICROVISION Namespaces
//----------------------------------------
using namespace microvision::common::sdk;
using namespace microvision::common::plugin_framework;

//========================================
// MICROVISION Objects
//----------------------------------------
const std::string appName = "movia";
microvision::common::logging::LoggerSPtr appLogger
    = microvision::common::logging::LogManager::getInstance().createLogger(appName);

//==============================================================================
//! \brief Listener class to receive and handle ldmi raw or icd mpl data.
//!
//! Depending on the received data a different converter to ROS PointCloud2 is used.
//------------------------------------------------------------------------------
class LdmiAndMplListener : public DataContainerListener<Scan2340, DataTypeId::DataType_LdmiRawFrame2353>,
                           public DataContainerListener<Scan2340, DataTypeId::DataType_LdmiRawFrame2354>,
                           public DataContainerListener<Image2404, DataTypeId::DataType_LdmiRawFrame2353>,
                           public DataContainerListener<Image2404, DataTypeId::DataType_LdmiRawFrame2354>,
                           public DataContainerListener<Scan2342, DataTypeId::DataType_Scan2342>,
                           public DataContainerListener<Image2404, DataTypeId::DataType_Image2404>
{
public:
    //========================================
    //!\brief Construct a new Listener object.
    //!
    //! param[in] _isTimeSynced  Defines whether sensor is in time sync.
    //                           If set to \c false data header contains time.now() of host.
    //----------------------------------------
    LdmiAndMplListener(rclcpp::Node& node,
                       rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcPublisher,
                       rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePublisher,
                       std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tfPublisher,
                       rclcpp::Clock::SharedPtr clock)
      : m_node{node},
        m_pointcloudPublisher{pcPublisher},
        m_imagePublisher{imagePublisher},
        m_transformPublisher{tfPublisher},
        m_rosClock{clock}
    {}

    //========================================
    //!\brief Destroy the Listener object
    //----------------------------------------
    virtual ~LdmiAndMplListener() = default;

    //========================================
    //! \brief Called on receiving a new Scan2340 data container.
    //!
    //! \param[in] scan           Shared pointer to an instance of Scan2340 that
    //!                           has been received.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //----------------------------------------
    void onData(std::shared_ptr<const Scan2340> scan, const ConfigurationPtr&) override
    {
        LOGDEBUG(appLogger,
                 "Scan for device ID: " << std::to_string(scan->getHeaderDeviceId()) << "/"
                                        << std::to_string(scan->getScannerInfo().getDeviceId()));

        // ROS Datatype
        sensor_msgs::msg::PointCloud2 msg;

        // convert MOVIA Scan to PointCloud2
        // If sensor is not time synced (set via parameter "sensor_timesynced") retrieve timestamp from host system
        rclcpp::Time scanTime = this->m_rosClock ? this->m_rosClock->now() : fromNtpTime(scan->getHeaderNtpTime());
        toPointCloud2(scanTime, scan, msg);

        // Publish PointCloud
        this->m_pointcloudPublisher->publish(msg);

        // TF frame
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp
            = this->m_rosClock ? this->m_rosClock->now() : fromNtpTime(scan->getHeaderNtpTime());
        transformStamped.header.frame_id = "world";
        transformStamped.child_frame_id  = "MicroVisionScanTF";
        // broadcast TF frames
        m_transformPublisher->sendTransform(transformStamped);

        LOGDEBUG(appLogger, "Pointcloud published with " << scan->getScanPoints().size() << " points!");
    }

    //========================================
    //! \brief Called on receiving a new Scan2342 data container.
    //!
    //! \param[in] scan           Shared pointer to an instance of Scan2342 that has been received.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //----------------------------------------
    void onData(std::shared_ptr<const Scan2342> scan, const ConfigurationPtr&) override
    {
        LOGDEBUG(appLogger,
                 "Scan for device ID: " << std::to_string(scan->getHeaderDeviceId()) << "/"
                                        << (scan->hasScannerInfo()
                                                ? (std::to_string(scan->getScannerInfo().getDataInfo().getDeviceId()))
                                                : "no scanner info"));

        // ROS Datatype
        sensor_msgs::msg::PointCloud2 msg;

        // convert MOVIA Scan to PointCloud2
        rclcpp::Time scanTime = this->m_rosClock ? this->m_rosClock->now() : fromNtpTime(scan->getHeaderNtpTime());
        toPointCloud2(scanTime, scan, msg);

        // Publish PointCloud
        this->m_pointcloudPublisher->publish(msg);

        // TF frame
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp
            = this->m_rosClock ? this->m_rosClock->now() : fromNtpTime(scan->getHeaderNtpTime());
        transformStamped.header.frame_id = "world";
        transformStamped.child_frame_id  = "MicroVisionScanTF";
        // broadcast TF frames
        this->m_transformPublisher->sendTransform(transformStamped);

        LOGDEBUG(appLogger, "Pointcloud published!");
    }

    //========================================
    //! \brief Called on receiving a new Image2404 data container.
    //!
    //! \param[in] image          Shared pointer to an instance of Image2404 that has been received.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //----------------------------------------
    void onData(std::shared_ptr<const Image2404> image, const ConfigurationPtr&) override
    {
        LOGDEBUG(appLogger,
                 "Image for device ID: " << std::to_string(image->getHeaderDeviceId()) << "/"
                                         << std::to_string(image->getDeviceId()));
        // ROS Datatype
        sensor_msgs::msg::Image msg;
        // Check image format and convert data to ROS2 datatype ff image is JPEG or MJPEG only
        auto imageFormat = image->getFormat();
        if ((imageFormat != image::ImageFormatIn2404::Jpeg) && (imageFormat != image::ImageFormatIn2404::Mjpeg)
            && (imageFormat != image::ImageFormatIn2404::Gray8))
        {
            LOGWARNING(appLogger, "Received unsupported image format!");
            return;
        }

        // Get raw image buffer
        const auto rawBuffer = image->getImageBuffer();
        if (!rawBuffer)
        {
            return;
        }

        if (imageFormat != image::ImageFormatIn2404::Gray8)
        {
            // Create new ring buffer for image data
            const unsigned int rgbBufferSize = (unsigned int)(image->getWidth() * image->getHeight() * 3);
            unsigned char* rgbBuffer         = new unsigned char[size_t(rgbBufferSize)];

            unsigned int width  = 0;
            unsigned int height = 0;
            // Fill rgbBuffer, width and height
            const int retCode
                = readJpegFromMemory(rgbBuffer,
                                     &width,
                                     &height,
                                     reinterpret_cast<const unsigned char*>(rawBuffer->getDataBuffer().data()),
                                     static_cast<uint32_t>(rawBuffer->getSize()));
            if (retCode != 1)
            {
                return;
            }

            // Set msg data
            msg.width  = width;
            msg.height = height;
            // Converting image data back to rgb8!
            msg.encoding = "rgb8";
            msg.step     = rgbBufferSize / msg.height;
            msg.data.clear();
            // Fill msg with image data
            for (uint32_t i = 0; i < rgbBufferSize; ++i)
            {
                msg.data.emplace_back(rgbBuffer[i]);
            }
            // Destroy rgbBuffer
            delete[] rgbBuffer;
        }
        else
        {
            // Set msg data
            msg.width                = image->getWidth();
            msg.height               = image->getHeight();
            msg.encoding             = "mono8";
            msg.step                 = msg.width;
            const auto rawBufferSize = image->getWidth() * image->getHeight();
            msg.data.resize(rawBufferSize);
            std::memcpy(msg.data.data(),
                        reinterpret_cast<const unsigned char*>(rawBuffer->getDataBuffer().data()),
                        rawBufferSize);
        }

        // Create new msg
        msg.header.frame_id = "MicroVisionScanTF";
        msg.header.stamp    = this->m_rosClock ? this->m_rosClock->now() : fromNtpTime(image->getHeaderNtpTime());

        // Publish converted data
        this->m_imagePublisher->publish(msg);
    }

    //========================================
    //! \brief Helper function from NtpTime to ROS2 time.
    //!
    //! \param[in] time  Source time.
    //! \returns  ROS2 time.
    //----------------------------------------
    static rclcpp::Time fromNtpTime(const NtpTime& time)
    {
        static const boost::posix_time::ptime epoch{boost::gregorian::date(1970, 1, 1)};
        const auto sinceEpoch = time.toPtime() - epoch;
        return rclcpp::Time(sinceEpoch.total_nanoseconds());
    }

private:
    //========================================
    //! \brief Convert SDK scan data container into PointCloud2 message
    //!
    //! \param[in]  timeStamp  Header time stamp.
    //! \param[in]  scan       Shared pointer to an instance of Scan2340 that has been received.
    //! \param[out] pcl        Returned PointCloud2 message
    //----------------------------------------
    void toPointCloud2(rclcpp::Time timeStamp, std::shared_ptr<const Scan2340> scan, sensor_msgs::msg::PointCloud2& pcl)
    {
        pcl.header.stamp    = timeStamp;
        pcl.header.frame_id = "MicroVisionScanTF";
        pcl.is_dense        = false;
        pcl.is_bigendian    = false;

        // define the pointcloud fields
        constexpr std::array<const char*, 12> fields{{{"x"},
                                                      {"y"},
                                                      {"z"},
                                                      {"echoID"},
                                                      {"existenceMeasure"},
                                                      {"radialDistance"},
                                                      {"horID"},
                                                      {"verID"},
                                                      {"Intensity"},
                                                      {"PulseWidth"},
                                                      {"BloomingMeasure"},
                                                      {"TimestampOffsetInUs"}}};

        pcl.fields.resize(fields.size());
        for (uint32_t idx = 0; idx < fields.size(); ++idx)
        {
            pcl.fields[idx].name     = fields[idx];
            pcl.fields[idx].offset   = idx * sizeof(float_t);
            pcl.fields[idx].datatype = sensor_msgs::msg::PointField::FLOAT32;
            pcl.fields[idx].count    = 1;
        }

        // size of one scanpoint
        pcl.point_step = pcl.fields.size() * sizeof(float_t);
        // number of scan points
        pcl.width = scan->getScanPoints().size();
        // size of all scanpoints * size of point
        pcl.data.resize(std::max(1U, pcl.width * pcl.point_step),
                        0x00U); // reserve enough space, empty pcl.data is not allowed
        pcl.row_step = pcl.data.size();
        // unordered point cloud format
        pcl.height = 1;

        // each scanpoint comes with a vector that contains additional properties
        // find information for faster access in the point loop
        const float* intensity       = nullptr;
        const float* pulsewidth      = nullptr;
        const float* bloomingmeasure = nullptr;
        for (const auto& info : scan->getScanPointInfos())
        {
            switch (info.getInformationType())
            {
            case ScanPointInfoListIn2340::InformationType::Intensity:
                intensity = reinterpret_cast<const float*>(info.getScanPointInformations().data());
                break;
            case ScanPointInfoListIn2340::InformationType::PulseWidth:
                pulsewidth = reinterpret_cast<const float*>(info.getScanPointInformations().data());
                break;
            case ScanPointInfoListIn2340::InformationType::BloomingMeasure:
                bloomingmeasure = reinterpret_cast<const float*>(info.getScanPointInformations().data());
            default:
                break;
            }
        }

        rcpputils::assert_true(intensity != nullptr);
        rcpputils::assert_true(pulsewidth != nullptr);
        rcpputils::assert_true(bloomingmeasure != nullptr);

        // now copy the points
        uint32_t numberOfPoints = 0;
        float* destination      = reinterpret_cast<float*>(pcl.data.data());
        for (uint32_t ptIdx = 0; ptIdx < scan->getScanPoints().size(); ++ptIdx)
        {
            const auto& pt              = scan->getScanPoints()[ptIdx];
            const auto existenceMeasure = pt.getExistenceMeasure();
            if (existenceMeasure >= existenceMeasureFilterValue)
            {
                *(destination + 0)  = pt.getPosition().getX();
                *(destination + 1)  = pt.getPosition().getY();
                *(destination + 2)  = pt.getPosition().getZ();
                *(destination + 3)  = static_cast<float>(pt.getEchoId());
                *(destination + 4)  = existenceMeasure;
                *(destination + 5)  = pt.getRadialDistance();
                *(destination + 6)  = static_cast<float>(pt.getHorizontalId());
                *(destination + 7)  = static_cast<float>(pt.getVerticalId());
                *(destination + 8)  = *intensity++;
                *(destination + 9)  = *pulsewidth++;
                *(destination + 10) = *bloomingmeasure++;
                *(destination + 11) = static_cast<float>(pt.getTimestampOffsetInUs());

                destination += fields.size();
                ++numberOfPoints;
            }
        }

        pcl.width = numberOfPoints;
        pcl.data.resize(std::max(1U, numberOfPoints * pcl.point_step), 0x00U); // cut down to sent number
    }

    //========================================
    //! \brief Convert SDK scan data container into PointCloud2 message
    //!
    //! \param[in]  timeStamp  Header time stamp.
    //! \param[in]  scan       Shared pointer to an instance of Scan2342 that has been received.
    //! \param[out] pcl        Returned PointCloud2 message
    //----------------------------------------
    void toPointCloud2(rclcpp::Time timeStamp, std::shared_ptr<const Scan2342> scan, sensor_msgs::msg::PointCloud2& pcl)
    {
        pcl.header.stamp    = timeStamp;
        pcl.header.frame_id = "MicroVisionScanTF";
        pcl.is_dense        = false;
        pcl.is_bigendian    = false;

        // define the pointcloud fields
        constexpr std::array<const char*, 11> fields{{{"x"},
                                                      {"y"},
                                                      {"z"},
                                                      {"echoID"},
                                                      {"existenceMeasure"},
                                                      {"radialDistance"},
                                                      {"horID"},
                                                      {"verID"},
                                                      {"Intensity"},
                                                      {"PulseWidth"},
                                                      {"TimestampOffsetInUs"}}};

        pcl.fields.resize(fields.size());
        for (uint32_t idx = 0; idx < fields.size(); ++idx)
        {
            pcl.fields[idx].name     = fields[idx];
            pcl.fields[idx].offset   = idx * sizeof(float_t);
            pcl.fields[idx].datatype = sensor_msgs::msg::PointField::FLOAT32;
            pcl.fields[idx].count    = 1;
        }

        // size of one scanpoint
        pcl.point_step = pcl.fields.size() * sizeof(float_t);
        // number of scan points per row
        const uint8_t numEchoes = 3; // depends on icd mpl!
        const auto& rowArray    = scan->getRows();
        const uint8_t numRows   = rowArray.size();
        pcl.width               = 128 * numEchoes * numRows;
        // size of all scanpoints * size of point
        pcl.data.resize(std::max(1U, pcl.width * pcl.point_step),
                        0x00U); // reserve enough space, empty pcl.data is not allowed
        pcl.row_step = pcl.data.size();
        // unordered point cloud format
        pcl.height = 1;

        // now copy the points
        float* destination               = reinterpret_cast<float*>(pcl.data.data());
        const auto& pixelDirectionsArray = scan->getScannerInfo().getDirections().getPixelDirections();
        auto pixelDirectionIter          = pixelDirectionsArray.cbegin();
        constexpr unit::Convert<unit::length::centimeter, unit::length::meter, float> meterToCentimeterConverter{};
        constexpr uint16_t uint16Max = std::numeric_limits<uint16_t>::max();
        constexpr float numLimit     = 1.0F / static_cast<float>(uint16Max - 1);

        uint32_t numberOfPoints        = 0;
        uint32_t numberOfInvalidPoints = 0;

        uint64_t scanStartTimestamp = rowArray.empty() ? 0 : rowArray.front().getTimestampStart().getMicroseconds();

        for (uint16_t rowId = 0; rowId < rowArray.size(); ++rowId)
        {
            const auto row              = rowArray[rowId];
            const auto& echoArraysInRow = row.getScanPoints();

            uint64_t timestampMicros = row.getTimestampStart().getMicroseconds();
            uint32_t offsetNanos     = row.getTimestampOffsetInNanoseconds();
            const auto rowTimeOffsetInUs = static_cast<float>(
                timestampMicros + (static_cast<uint64_t>(offsetNanos) / 1000) - scanStartTimestamp);

            for (uint8_t colId = 0; colId < echoArraysInRow.size(); ++colId)
            {
                const auto& echoArray      = echoArraysInRow[colId];
                const auto& pixelDirection = *(pixelDirectionIter++);
                const float azimuth        = pixelDirection.getX();
                const float elevation      = pixelDirection.getY();

                for (uint8_t echoId = 0; echoId < echoArray.size(); ++echoId)
                {
                    const auto& point = echoArray[echoId];

                    const auto existenceMeasure = static_cast<float>(point.getExistenceMeasure()) * numLimit;
                    if (existenceMeasure >= existenceMeasureFilterValue)
                    {
                        // calculate position
                        const float radDistance  = meterToCentimeterConverter(point.getRadialDistanceInCentimeter());
                        const float cosElevation = std::cos(elevation);
                        const float sinElevation = std::sin(elevation);
                        const float cosAzimuth   = std::cos(azimuth);
                        const float sinAzimuth   = std::sin(azimuth);

                        *(destination + 0) = radDistance * cosElevation * cosAzimuth; // azimuth 180 deg rotated
                        *(destination + 1) = radDistance * cosElevation * sinAzimuth; // elevation 90 deg rotated
                        *(destination + 2) = radDistance * sinElevation;
                        *(destination + 3) = static_cast<float>(echoId);
                        *(destination + 4) = existenceMeasure;
                        *(destination + 5) = radDistance;
                        *(destination + 6) = static_cast<float>(colId);
                        *(destination + 7) = static_cast<float>(rowId);
                        *(destination + 10) = rowTimeOffsetInUs;

                        // find invalid points
                        if ((point.getRadialDistanceInCentimeter() < uint16Max)
                            && (point.getExistenceMeasure() < uint16Max) && (point.getRadialDistanceInCentimeter() > 0))
                        {
                            *(destination + 8) = point.getIntensity();
                            *(destination + 9) = convertQ4_12ToFloat(point.getPulseWidth());
                            ++numberOfPoints;

                            destination += fields.size();
                        }
                        else
                        {
                            *(destination + 8) = 0.0F;
                            *(destination + 9) = 0.0F;
                            ++numberOfInvalidPoints;
                        }

                        if (static_cast<uint64_t>(destination - reinterpret_cast<float*>(pcl.data.data()))
                            > pcl.data.size())
                        {
                            // abort
                            throw std::runtime_error{"Too many points - some assumption about "
                                                     "number of columns or echoes is wrong!"};
                        }
                    }
                }
            }
        }

        pcl.width = numberOfPoints;
        pcl.data.resize(std::max(1U, numberOfPoints * pcl.point_step), 0x00U); // cut down to sent number

        LOGDEBUG(appLogger, "Points: " << numberOfPoints << "/" << numberOfInvalidPoints);
    }

private:
    rclcpp::Node& m_node;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_pointcloudPublisher;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_imagePublisher;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> m_transformPublisher;

    //!< nullptr if sensor is in time sync. Otherwise ros clock to use.
    rclcpp::Clock::SharedPtr m_rosClock;

    static constexpr float existenceMeasureFilterValue = 0.99;
}; // LdmiAndMplListener

//==============================================================================
//! \brief Collection of parameters which are used for the configuration of the MOVIA sensor.
//------------------------------------------------------------------------------
struct DeviceConfigParameters
{
    //========================================
    //! \brief Sensor Hardware Id.
    //--------------------------------------
    uint16_t m_hwId;

    //========================================
    //! \brief Multicast address.
    //!
    //! The MOVIA is streaming its point cloud data to a IPv6 multicast address
    //! (as UDP packages). In this regard it is important (not for MOVIA L)
    //! to include the scope ID of the used network interface
    //! in the address string. The scope ID is seperated from the
    //! IPv6 address with a '%' symbol, like in the example below:
    //!
    //! ff02::1be0:1%5
    //--------------------------------------
    std::string m_multicastAddress;

    //========================================
    //! \brief UDP port used for pointcloud data.
    //!
    //! 12345
    //--------------------------------------
    int m_port;

    //========================================
    //! \brief Remote IP.
    //!
    //! The IP address which is used for the control connection to the MOVIA
    //! sensor via TCP (using SOME/IP) to a client.
    //! \note This is not required for MOVIA L.
    //--------------------------------------
    std::string m_remoteIpAddress;

    //========================================
    //! \brief Image Source Value
    //!
    //! Characteristic used to color the pixels of the image stream
    //! Possible Values:
    //! - Intensity
    //! - Distance
    //! - PulseWidthRatio
    //! - AvgPeak
    //! - NoiseFloorValue
    //! - BinCountMean
    //! - BlindingFlag
    //! - EchoId
    //! - IntensityToNoiseFloorValue
    //! - IntensityToAvgPeak
    //--------------------------------------
    std::string m_imageSource{"Intensity"};

    //========================================
    //! \brief Image Rotation.
    //!
    //! Rotation of the image stream
    //! Possible Values:
    //! - 0Deg
    //! - 90Deg
    //! - 180Deg
    //! - 270Deg
    //--------------------------------------
    std::string m_imageRotation{"180Deg"};

    //========================================
    //! \brief Define whether sensor is in time sync.
    //--------------------------------------
    bool m_isSensorTimesynced{false};

    //========================================
    //! \brief Feed sensor device from PCAP file instead of live sensor connection.
    //--------------------------------------
    std::string m_pcapFile{};

    //========================================
    //! \brief Feed sensor device from IDC file instead of live sensor connection.
    //--------------------------------------
    std::string m_idcFile{};

    //========================================
    //! \brief Receive LDMI raw data from MOVIA L instead of icd_mpl.
    //--------------------------------------
    bool m_receiveLdmiRaw;
};

//==============================================================================
//! \brief Return a fully configured sensor device.
//! \param[in] param The parameters which are used to configure the sensor device.
//! \returns Configured sensor device.
//------------------------------------------------------------------------------
IdcDevicePtr getConfiguredDevice(const DeviceConfigParameters& param)
{
    IdcDevicePtr sensorDevice;

    // use plugin device
    // const std::string hwidName = (param.m_hwId == 0) ? "L" : ("HWID-" + std::to_string(param.m_hwId));
    // sensorDevice               = createMoviaDevice(hwidName,
    //                                  hwidName, // not used if "receive-ldmi-raw" is set to false - see below!
    //                                  param.m_multicastAddress,
    //                                  "",
    //                                  param.m_port,
    //                                  "",
    //                                  0,
    //                                  "",
    //                                  microvision::common::sdk::Optional<bool>(true),
    //                                  microvision::common::sdk::Optional<bool>(((param.m_hwId == 0) ? false : true)),
    //                                  microvision::common::sdk::Optional<bool>(((param.m_hwId == 0) ? false : true)));
    // If you want to use an older MOVIA sensor please add the following lines instead
    // using hardcoded configuration instead of the yaml file:
    sensorDevice = createMoviaDevice("INLSB-B1-1.3.0",
                                    "inraconfig10p5_11deg",
                                    param.m_multicastAddress,
                                    "",
                                    param.m_port,
                                    param.m_remoteIpAddress,
                                    55000,
                                    "",
                                    Optional<bool>{true},
                                    Optional<bool>{true});

    if (!sensorDevice)
    {
        throw std::runtime_error{"Creation of sensor device failed! Is the MOVIA "
                                 "device plugin loaded?"};
    }

    if (param.m_receiveLdmiRaw)
    {
        // enable raw ldmi data for MOVIA L
        auto deviceConfig = sensorDevice->getDeviceConfiguration();
        if (!deviceConfig->trySetValue("receive-ldmi-raw", true))
        {
            throw std::runtime_error{"Unable to set MOVIA L sensor device ldmi raw udp stream data format!"};
        }
    }

    // Get sensor device image configuration for modification
    auto sensorImporterConfigurations = sensorDevice->getImporterConfigurations("MoviaImage");
    if (sensorImporterConfigurations.size() >= 1)
    {
        auto imageConfig = sensorImporterConfigurations.front();
        if (imageConfig)
        {
            // Set image configuration values
            imageConfig->trySetValue("image_source_value", param.m_imageSource);
            imageConfig->trySetValue("image_rotation", param.m_imageRotation);
        }
        else
        {
            LOGWARNING(appLogger,
                       "Device does not support image configuration! Image topics might be missing or wrong!");
        }
    }
    else
    {
        LOGWARNING(appLogger, "Retrieval of device image configuration failed! No image topics will be available!");
    }

    return sensorDevice;
}

//==============================================================================
//! \brief Configure ROS node from yaml file.
//!
//! \param node                Ros node to be configured.
//! \param deviceConfigParams  Configuration parameters for MOVIA device to be filled from yaml.
//! \return Either \c true if successful or \c false if some required parameter was not set in the yaml.
//------------------------------------------------------------------------------
bool configureNode(rclcpp::Node& node, DeviceConfigParameters& deviceConfigParams)
{
    // Declare launch yaml file parameters
    node.declare_parameter("sensor_timesynced", false);
    node.declare_parameter("hwid", "12");
    node.declare_parameter(
        "multicast_ip",
        "ff02::1be0:1%10"); // scopeId is an adapter number, could also be %eth0.101 (linux requires number!)
    node.declare_parameter("port", 12345);
    node.declare_parameter("remote_ip",
                           "172.16.101.56"); // ip address of MOVIA sensor for tcp communication
    node.declare_parameter("pcap_file", "");
    node.declare_parameter("idc_file", "");
    node.declare_parameter("log_level", "Warning");
    node.declare_parameter("ldmi_raw", false);

    node.declare_parameter("image_source", deviceConfigParams.m_imageSource);
    node.declare_parameter("image_rotation", deviceConfigParams.m_imageRotation);

    std::string hwid{}, sdkLogLevel{};
    bool parameterError = false;

    if (!node.get_parameter("log_level", sdkLogLevel))
    {
        RCLCPP_WARN_STREAM(node.get_logger(), "Parameter missing: log_level");
    }

    // Get SDK logmanager and set loglevel. Modify for debugging as needed:
    microvision::common::logging::LogManager::getInstance().setDefaultLogLevel(sdkLogLevel.c_str());

    if (!node.get_parameter("hwid", hwid))
    {
        RCLCPP_ERROR_STREAM(node.get_logger(), "Parameter missing: hwid");
        return false;
    }

    if (hwid.compare("L") == 0)
    {
        // MOVIA L gets special configuration
        deviceConfigParams.m_hwId = 0; // no HWID
    }
    else
    {
        // Check and setup supported MOVIA sensor type/version
        deviceConfigParams.m_hwId = std::stoi(hwid);

        // Check supported MOVIA sensor connection parameters and setup sensor
        // configuration
        if (!node.get_parameter("remote_ip", deviceConfigParams.m_remoteIpAddress))
        {
            RCLCPP_ERROR_STREAM(node.get_logger(), "Parameter missing: remote_ip");
            return false;
        }
    }

    if (!node.get_parameter("multicast_ip", deviceConfigParams.m_multicastAddress))
    {
        RCLCPP_ERROR_STREAM(node.get_logger(), "Parameter missing: multicast_ip");
        return false;
    }

    if (!node.get_parameter_or("port", deviceConfigParams.m_port, 12345))
    {
        RCLCPP_WARN_STREAM(node.get_logger(), "Parameter missing: port");
    }

    // Check if parameter loading
    parameterError |= !node.get_parameter("sensor_timesynced", deviceConfigParams.m_isSensorTimesynced);

    parameterError |= !node.get_parameter("image_source", deviceConfigParams.m_imageSource);
    parameterError |= !node.get_parameter("image_rotation", deviceConfigParams.m_imageRotation);

    parameterError |= !node.get_parameter_or("pcap_file", deviceConfigParams.m_pcapFile, std::string());
    parameterError |= !node.get_parameter_or("idc_file", deviceConfigParams.m_idcFile, std::string());

    parameterError |= !node.get_parameter_or("ldmi_raw", deviceConfigParams.m_receiveLdmiRaw, false);

    if (parameterError)
    {
        RCLCPP_WARN_STREAM(node.get_logger(), "Unable to load optional configuration parameters for MOVIA sensor");
    }

    return true;
}

//==============================================================================
//! \brief Publish pointclouds and images from a MOVIA pcap recording.
//!
//! \param pcapFileName  Name of a pcap file to be used.
//! \param sensorDevice  MOVIA device to be fed from the pcap file instead of a live sensor connection.
//------------------------------------------------------------------------------
bool publishPcap(rclcpp::Node& node, const std::string& pcapFileName, const IdcDevicePtr& sensorDevice)
{
    // Create PCAP reader
    auto pcapReader = StreamReaderFactory::getInstance().createPackageReaderFromFile(pcapFileName);
    if (!pcapReader)
    {
        RCLCPP_ERROR_STREAM(node.get_logger(), "Creation of PCAP reader failed! Is the thirdparty-pcap-plugin loaded?");
        return false;
    }
    else
    {
        if (!pcapReader->open())
        {
            RCLCPP_ERROR_STREAM(node.get_logger(), "Unable to open PCAP file '" << pcapFileName << "'!");
            return false;
        }
        else
        {
            LOGINFO(appLogger, "Reading pcap file '" << pcapFileName << "' ...");
        }

        // Process packages from PCAP reader.
        std::size_t numberOfAcceptedPackages{};

        for (auto package = pcapReader->readFirstPackage(); package != nullptr; package = pcapReader->readNextPackage())
        {
            if (sensorDevice->processDataPackage(package))
            {
                ++numberOfAcceptedPackages;
            }
        }

        RCLCPP_INFO_STREAM(node.get_logger(),
                           "Number of accepted packages from pcap file '" << pcapFileName
                                                                          << "': " << numberOfAcceptedPackages);
    }

    return true;
}

//==============================================================================
//! \brief ROS node main application which listens for an MOVIA sensor device.
//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Init ROS2
    rclcpp::init(argc, argv);
    rclcpp::Node node(appName);

    // MOVIA sensor configuration
    DeviceConfigParameters deviceConfigParams;

    // configure node from yaml file parameters
    if (!configureNode(node, deviceConfigParams))
    {
        rclcpp::shutdown();
        exit(0);
    }
    else
    {
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcPublisher
            = node.create_publisher<sensor_msgs::msg::PointCloud2>("microvision/pointcloud", 10);
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePublisher
            = node.create_publisher<sensor_msgs::msg::Image>("microvision/image", 10);
        std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tfPublisher
            = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);

        // Load plugins.
        const std::string pluginDir{DEFAULT_PLUGIN_PATH};
        microvision::common::plugin_framework::plugin_loader::PluginLoaderConfig pluginConfig{};
        microvision::common::plugin_framework::plugin_loader::PluginLoader pluginLoader{pluginConfig};
        pluginLoader.addLibraryFolder(pluginDir); // Load MOVIA plugins
        pluginLoader.loadAllPlugins();

        // Print startup info
        if (deviceConfigParams.m_hwId > 0)
        {
            RCLCPP_INFO_STREAM(node.get_logger(),
                               std::endl
                                   << "###########################################################" << std::endl
                                   << "#  Welcome to ROS2 Node for MOVIA Sensor" << std::endl
                                   << "#    - HWID:              " << deviceConfigParams.m_hwId << std::endl
                                   << "#    - Multicast address: " << deviceConfigParams.m_multicastAddress << std::endl
                                   << "#    - Sensor address:    " << deviceConfigParams.m_remoteIpAddress << std::endl
                                   << "#    - Port:              " << deviceConfigParams.m_port << std::endl
                                   << (deviceConfigParams.m_receiveLdmiRaw ? "#    - Ldmi raw enabled\n" : "")
                                   << "###########################################################" << std::endl);
        }
        else
        {
            RCLCPP_INFO_STREAM(node.get_logger(),
                               std::endl
                                   << "###########################################################" << std::endl
                                   << "#  Welcome to ROS2 Node for MOVIA L Sensor" << std::endl
                                   << "#    - Multicast address: " << deviceConfigParams.m_multicastAddress << std::endl
                                   << "#    - Port:              " << deviceConfigParams.m_port << std::endl
                                   << (deviceConfigParams.m_receiveLdmiRaw ? "#    - Ldmi raw enabled\n" : "")
                                   << "###########################################################" << std::endl);
        }

        // Connect MOVIA device
        IdcDevicePtr sensorDevice;
        try
        {
            sensorDevice = getConfiguredDevice(deviceConfigParams);
            RCLCPP_INFO_STREAM(node.get_logger(), "Get configuration for MOVIA device");

            auto listener = std::make_shared<LdmiAndMplListener>(
                node,
                pcPublisher,
                imagePublisher,
                tfPublisher,
                deviceConfigParams.m_isSensorTimesynced ? nullptr : node.get_clock());
            sensorDevice->registerDataContainerListener(listener);
            RCLCPP_INFO_STREAM(node.get_logger(), "Register listener for MOVIA device");

            if (!deviceConfigParams.m_pcapFile.empty())
            {
                publishPcap(node, deviceConfigParams.m_pcapFile, sensorDevice);
            }
            else if (!deviceConfigParams.m_idcFile.empty())
            {
                // Read scans from idc file (no ldmi raw configured currently).
                IdcFileInput idc(deviceConfigParams.m_idcFile);
                if (idc.open())
                {
                    idc.registerDataContainerListener(listener);

                    idc.loopAndNotify();

                    idc.unregisterDataContainerListener(listener);
                }
                else
                {
                    LOGERROR(appLogger, deviceConfigParams.m_idcFile << " not open!");
                }
            }
            else
            {
                sensorDevice->connect();
                RCLCPP_INFO_STREAM(node.get_logger(), "Connect MOVIA device");

                if (!sensorDevice->isWorking())
                {
                    throw std::runtime_error("Error: sensorDevice not working!");
                }

                // Wait until connected
                {
                    rclcpp::Rate loop_rate(1);
                    while (rclcpp::ok() && sensorDevice->isWorking() && !sensorDevice->isConnected())
                    {
                        loop_rate.sleep();
                        RCLCPP_INFO_STREAM(node.get_logger(),
                                           "Waiting for MOVIA (" << deviceConfigParams.m_remoteIpAddress
                                                                 << ") to connect!");
                    }
                }

                // Spin node until shutdown, loop rate is 10HZ=100ms
                rclcpp::Rate loop_rate(10);
                while (rclcpp::ok() && sensorDevice->isConnected() && sensorDevice->isWorking())
                {
                    loop_rate.sleep();
                }
            }
        }
        catch (const std::exception& ex)
        {
            RCLCPP_ERROR_STREAM(node.get_logger(), ex.what());
            rcutils_reset_error();
        }

        try
        {
            // Disconnect MOVIA device
            sensorDevice->disconnect();
            RCLCPP_INFO_STREAM(node.get_logger(), "Disconnected MOVIA device!");
        }
        catch (const std::exception& ex)
        {
            RCLCPP_ERROR_STREAM(node.get_logger(), "Disconnect MOVIA device: " << ex.what());
            rcutils_reset_error();
        }
    }

    try
    {
        rclcpp::shutdown();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception during shutdown: " << e.what() << std::endl;
    }

    RCLCPP_INFO_STREAM(node.get_logger(), "All done!");

    return 0;
}
