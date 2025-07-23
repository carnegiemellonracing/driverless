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
//! \date June 18, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/image/ImageFormat.hpp>

#include <cstdint>
#include <memory>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace image {
//==============================================================================

//==============================================================================
//! \brief Buffer used to store image data.
//!
//! The ImageBuffer class supports images up to a total size 2^32 bytes.
//------------------------------------------------------------------------------
class ImageBuffer final
{
public:
    //========================================
    //! \brief Type of elements in the data buffer.
    //----------------------------------------
    using DataType = char;

    //========================================
    //! \brief Type of data buffer.
    //----------------------------------------
    using DataBufferType = std::vector<DataType>;

public:
    //========================================
    //! \brief Create an image buffer.
    //!
    //! \param[in] imageFormat   The format of the image in the buffer.
    //! \param[in] width         The width of the image in the buffer.
    //! \param[in] height        The height of the image in the buffer.
    //! \param[in] buffer        The buffer containing the image data.
    //! \param[in] bufferLength  The length of the image data in the buffer.
    //! \return The newly created image buffer.
    //----------------------------------------
    static std::shared_ptr<ImageBuffer> create(const ImageFormat& imageFormat,
                                               const uint32_t width,
                                               const uint32_t height,
                                               const DataType* const buffer,
                                               const std::size_t bufferLength);

    //========================================
    //! \brief Create an image buffer from data read from a JPEG file.
    //!
    //! \param[in] buffer        The buffer containing the JPEG data (including file headers).
    //! \param[in] bufferLength  The length of the data in the buffer.
    //! \return The newly created image buffer.
    //----------------------------------------
    static std::shared_ptr<ImageBuffer> createFromJpeg(const DataType* const buffer, const std::size_t bufferLength)
    {
        return create(ImageFormat(FileFormat::Jpeg, PixelFormat::Unknown), 0, 0, buffer, bufferLength);
    }

    //========================================
    //! \brief Create an image buffer from data read from a PNG file.
    //!
    //! \param[in] buffer        The buffer containing the PNG data (including file headers).
    //! \param[in] bufferLength  The length of the data in the buffer.
    //! \return The newly created image buffer.
    //----------------------------------------
    static std::shared_ptr<ImageBuffer> createFromPng(const DataType* const buffer, const std::size_t bufferLength)
    {
        return create(ImageFormat(FileFormat::Png, PixelFormat::Unknown), 0, 0, buffer, bufferLength);
    }

    //========================================
    //! \brief Create an image buffer from a block of pure image data.
    //!
    //! \param[in] pixelFormat   The pixel format of the image in the buffer.
    //! \param[in] width         The width of the image in the buffer.
    //! \param[in] height        The height of the image in the buffer.
    //! \param[in] buffer        The buffer containing the image data.
    //! \param[in] bufferLength  The length of the image data in the buffer.
    //! \return The newly created image buffer.
    //----------------------------------------
    static std::shared_ptr<ImageBuffer> createFromRaw(const PixelFormat& pixelFormat,
                                                      const uint32_t width,
                                                      const uint32_t height,
                                                      const DataType* const buffer,
                                                      const std::size_t bufferLength)
    {
        return create(ImageFormat(FileFormat::Raw, pixelFormat), width, height, buffer, bufferLength);
    }

public:
    //========================================
    //!\brief Create an empty buffer.
    //----------------------------------------
    ImageBuffer() = default;

    //========================================
    //! \brief Copy constructor.
    //!
    //! \param[in] other  ImageBuffer to be cloned into this object.
    //----------------------------------------
    ImageBuffer(const ImageBuffer& other) = default;

    //========================================
    //! \brief Assigment operator.
    //!
    //! \param[in] other  ImageBuffer to be cloned into this object.
    //----------------------------------------
    ImageBuffer& operator=(const ImageBuffer& other) = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~ImageBuffer() = default;

public:
    //========================================
    //! \brief Get the image format.
    //!
    //! \return Format of the image in the buffer.
    //----------------------------------------
    const ImageFormat& getImageFormat() const { return m_imageFormat; }

    //========================================
    //! \brief Get the width of the image in the buffer in pixels.
    //!
    //! \return Width of the image in the buffer in pixels.
    //----------------------------------------
    uint32_t getWidth() const { return m_width; }

    //========================================
    //! \brief Get the height of the image in the buffer in pixels.
    //!
    //! \return Height of the image in the buffer in pixels.
    //----------------------------------------
    uint32_t getHeight() const { return m_height; }

    //========================================
    //! \brief Get the total size of the image data in bytes.
    //!
    //! \return The total size of the image data in bytes.
    //!
    //! The total size might not be equal to width*height*bytes_per_pixel (e.g. for compressed images or because of
    //! strides).
    //----------------------------------------
    size_t getSize() const { return m_buffer.size(); }

    //========================================
    //! \brief Get the buffer containing the image data.
    //!
    //! \return Buffer containing the image data.
    //----------------------------------------
    const DataBufferType& getDataBuffer() const { return m_buffer; }

    //========================================
    //! \brief Get the buffer containing the image data.
    //!
    //! \return Buffer containing the image data.
    //----------------------------------------
    DataBufferType& getDataBuffer() { return m_buffer; }

public:
    //========================================
    //! \brief Fill the image buffer with new data.
    //!
    //! \param[in] imageFormat   The format of the image in the buffer.
    //! \param[in] width         The width of the image in the buffer.
    //! \param[in] height        The height of the image in the buffer.
    //! \param[in] buffer        The buffer containing the image data.
    //! \param[in] bufferLength  The length of the image data in the buffer.
    //----------------------------------------
    void set(const ImageFormat& imageFormat,
             const uint32_t width,
             const uint32_t height,
             const DataType* const buffer,
             const std::size_t bufferLength);

    //========================================
    //! \brief Fill the image buffer with data read from a JPEG file.
    //!
    //! \param[in] buffer        The buffer containing the JPEG data (including file headers).
    //! \param[in] bufferLength  The length of the data in the buffer.
    //----------------------------------------
    void setFromJpeg(const DataType* const buffer, const std::size_t bufferLength)
    {
        set(ImageFormat(FileFormat::Jpeg, PixelFormat::Unknown), 0, 0, buffer, bufferLength);
    }

    //========================================
    //! \brief Fill the image buffer with data read from a PNG file
    //!
    //! \param[in] buffer        The buffer containing the PNG data (including file headers).
    //! \param[in] bufferLength  The length of the data in the buffer.
    //----------------------------------------
    void setFromPng(const DataType* const buffer, const std::size_t bufferLength)
    {
        set(ImageFormat(FileFormat::Png, PixelFormat::Unknown), 0, 0, buffer, bufferLength);
    }

    //========================================
    //! \brief Fill the image buffer with a block of pure image data.
    //!
    //! \param[in] pixelFormat   The pixel format of the image in the buffer.
    //! \param[in] width         The width of the image in the buffer.
    //! \param[in] height        The height of the image in the buffer.
    //! \param[in] buffer        The buffer containing the image data.
    //! \param[in] bufferLength  The length of the image data in the buffer.
    //----------------------------------------
    void setFromRaw(const PixelFormat& pixelFormat,
                    const uint32_t width,
                    const uint32_t height,
                    const DataType* const buffer,
                    const std::size_t bufferLength)
    {
        set(ImageFormat(FileFormat::Raw, pixelFormat), width, height, buffer, bufferLength);
    }

    //========================================
    //! \brief Preset the image buffer with the given information and reserve space to hold the image data.
    //!
    //! \param[in] imageFormat  The format of the image to be stored in the buffer.
    //! \param[in] width        The width of the image to be stored in the buffer.
    //! \param[in] height       The height of the image to be stored in the buffer.
    //! \param[in] imageSize    The length of the image data to be stored in the buffer.
    //----------------------------------------
    void
    preset(const ImageFormat& imageFormat, const uint32_t width, const uint32_t height, const std::size_t imageSize);

private:
    ImageFormat m_imageFormat{};
    uint32_t m_width{0};
    uint32_t m_height{0};
    DataBufferType m_buffer{};
}; // ImageBuffer

//==============================================================================

using ImageBufferPtr = std::shared_ptr<ImageBuffer>;

//==============================================================================

bool operator==(const ImageBuffer& lhs, const ImageBuffer& rhs);
inline bool operator!=(const ImageBuffer& lhs, const ImageBuffer& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace image
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
