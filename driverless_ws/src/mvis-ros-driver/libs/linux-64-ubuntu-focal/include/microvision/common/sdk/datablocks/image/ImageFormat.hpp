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
//! \date Mar 07, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/image/FileFormat.hpp>
#include <microvision/common/sdk/datablocks/image/PixelFormat.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace image {
//==============================================================================

//==============================================================================
//!\brief Format parameter of an image.
//------------------------------------------------------------------------------
class ImageFormat final
{
public:
    //==============================================================================
    //!\brief Converts the file format to a human readable string.
    //!
    //!\param[in] format  File format id to be converted into a string.
    //------------------------------------------------------------------------------
    static std::string formatToString(const FileFormat fileFormat);

    //==============================================================================
    //!\brief Converts the pixel format to a human readable string.
    //!
    //!\param[in] format  Pixel format id to be converted into a string.
    //------------------------------------------------------------------------------
    static std::string formatToString(const PixelFormat pixelFormat);

    //==============================================================================
    //!\brief Converts the image format to a human readable string.
    //!
    //!\param[in] format  Image format id to be converted into a string.
    //------------------------------------------------------------------------------
    static std::string formatToString(const ImageFormat format);

public: // constructors, destructors
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    ImageFormat() = default;

    //========================================
    //! \brief Constructor.
    //!
    //! \param[in] fileFormat   The file format within this image format.
    //! \param[in] pixelFormat  The pixel format within this image format.
    //----------------------------------------
    ImageFormat(FileFormat fileFormat, PixelFormat pixelFormat) : m_fileFormat(fileFormat), m_pixelFormat(pixelFormat)
    {}

public: // getter
    //========================================
    //! \brief Returns the file format.
    //!
    //! \return The file format of this image format.
    //----------------------------------------
    FileFormat getFileFormat() const { return m_fileFormat; }

    //========================================
    //! \brief Returns the pixel format.
    //!
    //! \return The pixel format of this image format.
    //----------------------------------------
    PixelFormat getPixelFormat() const { return m_pixelFormat; }

    //========================================
    //! \brief Get the number of bytes per pixels.
    //!
    //! \return Number of pixels in the buffer or -1 if the file format is compressed or the pixel format is unknown.
    //----------------------------------------
    int32_t getBytesPerPixels() const;

public: // setter
    //========================================
    //! \brief Set the file format within this image format.
    //!
    //! \param[in] fileFormat  The new file format.
    //----------------------------------------
    void setFileFormat(FileFormat fileFormat) { m_fileFormat = fileFormat; }

    //========================================
    //! \brief Set the pixel format within this image format.
    //!
    //! \param[in] pixelFormat  The new pixel format.
    //----------------------------------------
    void setPixelFormat(PixelFormat pixelFormat) { m_pixelFormat = pixelFormat; }

private:
    FileFormat m_fileFormat{FileFormat::Unknown};
    PixelFormat m_pixelFormat{PixelFormat::Unknown};
}; // ImageFormat

//==============================================================================
//==============================================================================

inline bool operator==(const ImageFormat& lhs, const ImageFormat& rhs)
{
    return (lhs.getFileFormat() == rhs.getFileFormat()) && (lhs.getPixelFormat() == rhs.getPixelFormat());
}

//==============================================================================

inline bool operator!=(const ImageFormat& lhs, const ImageFormat& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace image
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
