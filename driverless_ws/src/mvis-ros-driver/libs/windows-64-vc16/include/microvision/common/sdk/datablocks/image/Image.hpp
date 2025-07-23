//==============================================================================
//! \file
//!
//!\verbatim
//! General data container image
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//!
//! Created by
//! \date June 18, 2018
//!\endverbatim
//------------------------------------------------------------------------------
//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/image/ImageBuffer.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/MountingPosition.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/Vector2.hpp>

#include <microvision/common/sdk/io.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief General data container for images
//------------------------------------------------------------------------------
class Image final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Type of data stored in the image buffer.
    //----------------------------------------
    using DataType = image::ImageBuffer::DataType;

    //========================================
    //! \brief The coordinate system in which the
    //!        mounting position is given.
    //----------------------------------------
    enum class MountingPositionCoordinateSystem : uint8_t
    {
        SensorCoordinateSystem = 0, //!< ISO 8855, x forward. E.g. used for intensity images.
        CameraCoordinateSystem = 1 //!< z forward, E.g. used for camera images.
    };

    //========================================
    //! \brief Five polynomial parameters for image distortion.
    //!
    //! The parameters are \f$k_1\f$, \f$k_2\f$, \f$p_1\f$, \f$p_2\f$ and \f$k_3\f$.
    //----------------------------------------
    using DistortionParams = std::array<double, 5>;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.Image"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Image() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~Image() = default;

    //========================================
    //! \brief Assignment operator.
    //!
    //! \param[in] other  object to be assigned
    //! \return this
    //----------------------------------------
    Image& operator=(const Image& other) = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Copies image data from source into this image object.
    //!
    //! This function copies an image from the source buffer
    //! to this object.
    //!
    //! \param[in] imageFormat   Format of the image resides in \a buffer.
    //! \param[in] buffer        Source buffer, where the original image source resides.
    //! \param[in] length        Size of image in bytes.
    //! \param[in] width         Width of the image.
    //! \param[in] height        Height of the image.
    //! \param[in] microseconds  Number of microseconds passed since powering up the device.
    //! \param[in] timestamp     Time, the image was taken.
    //----------------------------------------
    void setImage(const image::ImageFormat imageFormat,
                  const DataType* const buffer,
                  const uint32_t length,
                  const uint32_t width,
                  const uint32_t height,
                  const uint32_t microseconds,
                  const microvision::common::sdk::NtpTime timestamp);

public:
    //========================================
    //! \brief Returns the format of the image in the buffer.
    //!
    //! \return Format of the image in the buffer.
    //----------------------------------------
    image::ImageFormat getFormat() const { return m_imageBuffer->getImageFormat(); }

    //========================================
    //! \brief Get the number of elapsed microseconds since the start
    //! of the device that created this Image DataContainer.
    //! \return The number of elapsed microseconds since the start of the device.
    //----------------------------------------
    uint32_t getUpTimeInMicroseconds() const { return m_upTimeInMicroseconds; }

    //========================================
    //! \brief Get the timestamp of when this Image has been generated.
    //! \return The timestamp of when this Image has been generated.
    //----------------------------------------
    NtpTime getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief Get the device id of the camera that generated this image.
    //!
    //! \return Device id of camera that generated this image.
    //----------------------------------------
    uint8_t getDeviceId() const { return m_deviceId; }

    //========================================
    //! \brief Get the coordinate system of the mounting position.
    //! \return The coordinate system of the mounting position.
    //----------------------------------------
    MountingPositionCoordinateSystem getMountingPositionCoordinateSystem() const
    {
        return m_mountingPositionCoordinateSystem;
    }

    //========================================
    //! \brief Get the cameras mounting position.
    //!
    //! In case one ore more attributes of the returned
    //! MountingPosition<float> are set to NaN all data are invalid.
    //!
    //! \return Cameras mounting position.
    //----------------------------------------
    MountingPosition<float> getCamMountingPosition() const { return m_mountingPosition; }

    //========================================
    //! \brief Get the horizontal opening angle of the camera.
    //!
    //! Returns the horizontal opening angle of the camera.
    //! Note that the complete position is valid ONLY if this angle is not NaN.
    //!
    //! \return Horizontal opening angle of the camera, in [rad].
    //----------------------------------------
    double getCamHorizontalOpeningAngle() const { return m_horizontalOpeningAngle; }

    //========================================
    //! \brief Get the vertical opening angle of the camera.
    //!
    //! Returns the vertical opening angle of the camera.
    //! Note that the complete position is valid ONLY if this angle is not NaN.
    //!
    //! \return Vertical opening angle of the camera, in [rad].
    //----------------------------------------
    double getCamVerticalOpeningAngle() const { return m_verticalOpeningAngle; }

    //========================================
    //! Returns camera parameters used when this image was taken.
    //!
    //! \param[out] focalLength     Focal length for the camera that took the image.
    //! \param[out] principlePoint  Center point of the camera that took the image.
    //! \param[out] distortion      Camera lens distortion parameters.
    //----------------------------------------
    void getCameraIntrinsics(Vector2<double>& focalLength,
                             Vector2<double>& principlePoint,
                             DistortionParams& distortion) const
    {
        focalLength    = m_focalLength;
        principlePoint = m_principlePoint;
        distortion     = m_lensDistortion;
    }

    //========================================
    //! \brief Return the width (number of pixels per row) of the image.
    //! \return Number of pixels per row.
    //----------------------------------------
    uint32_t getWidth() const { return m_imageBuffer->getWidth(); }

    //========================================
    //! \brief Return the height (number of rows) of the image.
    //! \return Number of rows.
    //----------------------------------------
    uint32_t getHeight() const { return m_imageBuffer->getHeight(); }

    //========================================
    //! \brief Return the size of the image in the buffer in bytes.
    //! \return Size of the image in the buffer in bytes.
    //----------------------------------------
    uint32_t getImageSize() const { return static_cast<uint32_t>(m_imageBuffer->getSize()); }

    //========================================
    //! \brief Get the buffer with image data and parameters.
    //! \return Buffer with image data and parameters.
    //----------------------------------------
    image::ImageBufferPtr getImageBuffer() const { return m_imageBuffer; }

public:
    //========================================
    //! \brief Set the number of elapsed microseconds.
    //!
    //! Since the start of the device that created this Image DataContainer.
    //!
    //! \param[in] newMicroseconds  New number of elapsed microseconds.
    //----------------------------------------
    void setMicroseconds(const uint32_t newMicroseconds) { m_upTimeInMicroseconds = newMicroseconds; }

    //========================================
    //! \brief Set the timestamp when this image has been generated.
    //!
    //! \param[in] newTimestamp  New timestamp.
    //----------------------------------------
    void setTimestamp(const NtpTime newTimestamp) { m_timestamp = newTimestamp; }

    //========================================
    //! \brief Set the device id of the camera for this image.
    //!
    //! \param[in] newDeviceId  New device id of camera for this image.
    //----------------------------------------
    void setDeviceId(const uint8_t newDeviceId) { m_deviceId = newDeviceId; }

    //========================================
    //! \brief Set the coordinate system of the mounting position.
    //! \param[in] newMountingPositionCoordinateSytem  New coordinate system of the mounting position.
    //----------------------------------------
    void setMountingPositionCoordinateSystem(const MountingPositionCoordinateSystem newMountingPositionCoordinateSystem)
    {
        m_mountingPositionCoordinateSystem = newMountingPositionCoordinateSystem;
    }

    //========================================
    //! \brief Set the mounting position of the camera for this image.
    //!
    //! \param[in] pos  Mounting position of the camera.
    //! \note The mounting position is relative to the vehicle/reference coordinate system.
    //----------------------------------------
    void setCamMountingPosition(const MountingPosition<float>& pos) { m_mountingPosition = pos; }

    //========================================
    //! \brief Set the horizontal opening angle of the camera took this image.
    //!
    //! \param[in] hOpeningAngle  Horizontal opening (view) angle in [rad].
    //----------------------------------------
    void setCamHorizontalOpeningAngle(const double hOpeningAngle) { m_horizontalOpeningAngle = hOpeningAngle; }

    //========================================
    //! \brief Set the vertical opening angle of the camera took this image.
    //!
    //! \param[in] vOpeningAngle  Vertical opening (view) angle in [rad].
    //----------------------------------------
    void setCamVerticalOpeningAngle(const double vOpeningAngle) { m_verticalOpeningAngle = vOpeningAngle; }

    //========================================
    //! \brief Sets camera intrinsics.
    //!
    //! \param[in] focalLength     Focal length for the camera that took the image.
    //! \param[in] principlePoint  Center point of the camera that took the image.
    //! \param[in] distortion      Distortion parameters of the camera that took the image.
    //----------------------------------------
    void setCameraIntrinsics(const Vector2<double>& focalLength,
                             const Vector2<double>& principlePoint,
                             const DistortionParams& distortion)
    {
        m_focalLength    = focalLength;
        m_principlePoint = principlePoint;
        m_lensDistortion = distortion;
    }

    //========================================
    //! \brief Set the buffer with image data and parameters.
    //!
    //! \param[in] imageBuffer  The new buffer with image data and parameters.
    //----------------------------------------
    void setImageBuffer(const image::ImageBufferPtr& imageBuffer) { m_imageBuffer = imageBuffer; }

private:
    //========================================
    //! \brief Reserve space in the image buffer to hold the given number of bytes and set other image parameter.
    //!
    //! \param[in] imageFormat  Format of the image to store into the buffer.
    //! \param[in] imageSize    Size of the image data to store into the buffer.
    //! \param[in] width        Width of the image to store into the buffer.
    //! \param[in] height       Height of the image to store into the buffer.
    //----------------------------------------
    void presetImageBuffer(const image::ImageFormat imageFormat,
                           const uint32_t imageSize,
                           const uint16_t width,
                           const uint16_t height)
    {
        m_imageBuffer->preset(imageFormat, width, height, imageSize);
    }

private: // attributes
    //========================================
    //! \brief Passed microseconds since start.
    //----------------------------------------
    uint32_t m_upTimeInMicroseconds{0};

    //========================================
    //! \brief Timestamp of this DataContainer.
    //----------------------------------------
    microvision::common::sdk::NtpTime m_timestamp{};

    //========================================
    //! \brief Device id of the origin of this image.
    //----------------------------------------
    uint8_t m_deviceId{0};

    //========================================
    //! \brief Coordinate system used for mounting position.
    //----------------------------------------
    MountingPositionCoordinateSystem m_mountingPositionCoordinateSystem{
        MountingPositionCoordinateSystem::SensorCoordinateSystem};

    //========================================
    //! \brief Mounting position of the camera this image has been taken with.
    //----------------------------------------
    MountingPosition<float> m_mountingPosition{};

    //========================================
    //! \brief Horizontal opening angle of the camera this image has been taken with.
    //----------------------------------------
    double m_horizontalOpeningAngle{NaN_double};

    //========================================
    //! \brief Vertical opening angle of the camera this image has been taken with.
    //----------------------------------------
    double m_verticalOpeningAngle{NaN_double};

    //========================================
    //! \brief Camera intrinsics focal length.
    //----------------------------------------
    Vector2<double> m_focalLength;

    //========================================
    //! \brief Camera intrinsics principle/optical center point.
    //----------------------------------------
    Vector2<double> m_principlePoint;

    //========================================
    //! \brief Camera intrinsics lens distortion polynomial parameters.
    //!
    //! Describes lens distortion relative to camera lens center point.
    //! k1, k2, t1, t2, k3
    //----------------------------------------
    DistortionParams m_lensDistortion{{0.0}};

    //========================================
    //! \brief Format of the image stored in m_imageBuffer.
    //----------------------------------------
    image::ImageBufferPtr m_imageBuffer{std::make_shared<image::ImageBuffer>()};
}; // Image

//==============================================================================

bool operator==(const Image& lhs, const Image& rhs);
inline bool operator!=(const Image& lhs, const Image& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
