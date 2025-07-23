//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 6th, 2019
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/image/special/ImageFormatIn2404.hpp>
#include <microvision/common/sdk/datablocks/image/ImageBuffer.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/MountingPosition.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/Vector2.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Specialized data container holding an image used
//!        for video frame images (0x2404).
//!
//! This class holds one frame of a video usually recorded
//! by the reference camera connected to the ECU. The frame
//! can be set using either set-function. Also camera intrinsics
//! are stored in this data container.
//! The functions copy the content of the passed buffer to
//! the class' own buffer.
//!
//! Note: See https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
//!       for information about the camera lens distortion (radial/tangential).
//!       Only base radial and tangential distortion values k1, k2, p1, p3 (or t1,
//!       t2 depending on naming) are stored in image 2404.
//!
//! General data type: \ref microvision::common::sdk::Image
//------------------------------------------------------------------------------
class Image2404 : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //!\brief Type of data stored in the image buffer.
    //----------------------------------------
    using DataType = image::ImageBuffer::DataType;

    //========================================
    //! \brief Four polynom parameters for image distortion.
    //----------------------------------------
    using DistortionParams = std::array<double, 4>;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.image2404"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Default Constructor.
    //----------------------------------------
    Image2404() = default;

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    virtual ~Image2404() = default;

    //========================================
    //! \brief Assignment operator.
    //!
    //! \param[in] other  Image to be assigned to this one.
    //! \return This image.
    //----------------------------------------
    Image2404& operator=(const Image2404& other) = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Copies image data from source into this image object.
    //!
    //! This function copies an image from the source buffer to this object.
    //!
    //! \param[in] imageFormat           Format of the image resides in \a buffer.
    //! \param[in] buffer                Source buffer, where the original image source resides.
    //! \param[in] length                Size of image in bytes.
    //! \param[in] widthInPixel          Width of the image in pixels per row.
    //! \param[in] heightInPixel         Height of the image in rows of pixels.
    //! \param[in] upTimeInMicroseconds  Number of microseconds passed since powering up the device.
    //! \param[in] timestamp             The time when the image was taken.
    //----------------------------------------
    void setImage(const image::ImageFormatIn2404 imageFormat,
                  const DataType* const buffer,
                  const uint32_t length,
                  const uint16_t widthInPixel,
                  const uint16_t heightInPixel,
                  const uint32_t upTimeInMicroseconds,
                  const microvision::common::sdk::NtpTime timestamp);

public:
    //========================================
    //! \brief Returns the format of the image in the buffer.
    //! \return Format of the image in the buffer.
    //----------------------------------------
    image::ImageFormatIn2404 getFormat() const { return getImageFormatIn2404(m_imageBuffer->getImageFormat()); }

    //========================================
    //! \brief Get the number of elapsed microseconds since the
    //!        start of the device that created this Image2403 DataContainer.
    //!
    //! \return The number of elapsed microseconds since the
    //!        start of the device
    //----------------------------------------
    uint32_t getUpTimeInMicroseconds() const { return m_upTimeInMicroseconds; }

    //========================================
    //! \brief Get the timestamp of when this Image2404 has been generated.
    //! \return The timestamp of when this Image2404 has been generated.
    //----------------------------------------
    NtpTime getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief Get the device id of the camera that generated this image.
    //! \return Device id of camera that generated this image.
    //----------------------------------------
    uint8_t getDeviceId() const { return m_deviceId; }

    //========================================
    //! \brief Get the camera mounting position.
    //!
    //! In case one ore more attributes of the returned
    //! MountingPosition<float> are set to NaN all data are invalid.
    //!
    //! \return Cameras mounting position.
    //----------------------------------------
    const MountingPosition<float>& getCamMountingPosition() const { return m_mountingPosition; }

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
    //! \param[out] focalLength    Focal length for the camera that took the image.
    //! \param[out] principlePoint Center point of the camera that took the image.
    //! \param[out] distortion     Camera lens distortion parameters k1, k2, t1, t2
    //!                            for the taylor series expansion.
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
    uint16_t getWidth() const { return static_cast<uint16_t>(m_imageBuffer->getWidth()); }

    //========================================
    //! \brief Return the height (number of rows) of the image.
    //! \return Number of rows.
    //----------------------------------------
    uint16_t getHeight() const { return static_cast<uint16_t>(m_imageBuffer->getHeight()); }

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

    //========================================
    //! \brief Convert an image format in container 0x2403 to a general image format.
    //!
    //! \param[in] imageFormat  The image format in container 0x2403.
    //! \return The general image format.
    //----------------------------------------
    static image::ImageFormat getGeneralImageFormat(const image::ImageFormatIn2404 imageFormat);

    //========================================
    //! \brief Convert a general image format to an image format in container 0x2403.
    //!
    //! \param[in] imageFormat  The general image format.
    //! \return The image format in container 0x2403.
    //----------------------------------------
    static image::ImageFormatIn2404 getImageFormatIn2404(const image::ImageFormat imageFormat);

public:
    //========================================
    //! \brief Set the number of elapsed microseconds since the start of the device
    //!        which created this Image2404 DataContainer.
    //! \param[in] newUpTimeInMicroseconds  New number of elapsed microseconds.
    //----------------------------------------
    void setUpTimeInMicroseconds(const uint32_t newUpTimeInMicroseconds)
    {
        m_upTimeInMicroseconds = newUpTimeInMicroseconds;
    }

    //========================================
    //! \brief Set the timestamp when this image has been generated.
    //! \param[in] newTimestamp  New timestamp.
    //----------------------------------------
    void setTimestamp(const NtpTime newTimestamp) { m_timestamp = newTimestamp; }

    //========================================
    //! \brief Set the device id of the camera for this image.
    //! \param[in] newDeviceId  New device id of camera for this image.
    //----------------------------------------
    void setDeviceId(const uint8_t newDeviceId) { m_deviceId = newDeviceId; }

    //========================================
    //! \brief Set the mounting position of the camera for this image.
    //!
    //! \note The mounting position is relative to the vehicle/reference coordinate system.
    //!
    //! \param[in] pos  Mounting position of the camera.
    //----------------------------------------
    void setCamMountingPosition(const MountingPosition<float>& pos) { m_mountingPosition = pos; }

    //========================================
    //! \brief Set the horizontal opening angle of the camera took this image.
    //! \param[in] hOpeningAngle  Horizontal opening (view) angle in [rad].
    //----------------------------------------
    void setCamHorizontalOpeningAngle(const double hOpeningAngle) { m_horizontalOpeningAngle = hOpeningAngle; }

    //========================================
    //! \brief Set the vertical opening angle of the camera took this image.
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
    void presetImageBuffer(const image::ImageFormatIn2404 imageFormat,
                           const uint32_t imageSize,
                           const uint16_t width,
                           const uint16_t height)
    {
        m_imageBuffer->preset(getGeneralImageFormat(imageFormat), width, height, imageSize);
    }

private:
    static constexpr const char* loggerId = "microvision::common::sdk::Image2404";
    static microvision::common::logging::LoggerSPtr logger;

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
    //! \brief Mounting position of the camera this image has been taken with.
    //----------------------------------------
    MountingPosition<float> m_mountingPosition{};

    //========================================
    // !\brief Horizontal opening angle of the camera this image has been taken with.
    //----------------------------------------
    double m_horizontalOpeningAngle{NaN_double};

    //========================================
    //! \brief Vertical opening angle of the camera this image has been taken with.
    //----------------------------------------
    double m_verticalOpeningAngle{NaN_double};

    //========================================
    //!\brief Camera intrinsics focal length.
    //----------------------------------------
    Vector2<double> m_focalLength;

    //========================================
    //!\brief Camera intrinsics principle/optical center point.
    //----------------------------------------
    Vector2<double> m_principlePoint;

    //========================================
    //!\brief Camera intrinsics lens distortion polynomial parameters.
    //!
    //! Describes lens distortion relative to camera lens center point.
    //! k1, k2, t1, t2
    //----------------------------------------
    DistortionParams m_lensDistortion{{0.0}};

    //========================================
    //! \brief Format of the image stored in m_imageBuffer.
    //----------------------------------------
    image::ImageBufferPtr m_imageBuffer{std::make_shared<image::ImageBuffer>()};
}; // Image2404

//==============================================================================

//==============================================================================
//! \ brief Compare two images for equality.
//!
//! \param[in] lhs  First image to compare.
//! \param[in] rhs  Second image to compare.
//! \return \c True if images are equal, \c false if not.
//------------------------------------------------------------------------------
bool operator==(const Image2404& lhs, const Image2404& rhs);

//==============================================================================
//! \ brief Check if two images are not equal.
//!
//! \param[in] lhs  First image to compare.
//! \param[in] rhs  Second image to compare.
//! \return \c True if images are different, \c false if not.
//------------------------------------------------------------------------------
inline bool operator!=(const Image2404& lhs, const Image2404& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
