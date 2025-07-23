//==============================================================================
//! \file
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io.hpp> //_prototypes.hpp>

#include <array>
#include <type_traits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief RGBA color value.
//! \tparam ColorValue  Can be uint8_t or float.
//!
//! * For ColorValue being uint8_t, the range for each component is 0 to 255.
//! * For ColorValue being float, the range for each component is 0 to 1.
//------------------------------------------------------------------------------
template<typename ColorValue>
class RgbaColor final
{
public:
    //========================================
    //! \brief Array with the 4 color components rgba.
    //----------------------------------------
    using ColorValueArray = std::array<ColorValue, 4>;

public:
    //========================================
    //! \brief Get the serialized size.
    //! \return Return the serialized size.
    //----------------------------------------
    static constexpr std::streamsize getSerializedSize()
    {
        static_assert(std::is_same<uint8_t, ColorValue>::value || std::is_same<float, ColorValue>::value,
                      "Only uint8_t and float are allowed here");
        return static_cast<std::streamsize>(4 * sizeof(ColorValue));
    }

private:
    //========================================
    //! \brief Array index of the red component value.
    //----------------------------------------
    static constexpr uint8_t redIndex = 0;

    //========================================
    //! \brief Array index of the green component value.
    //----------------------------------------
    static constexpr uint8_t greenIndex = 1;

    //========================================
    //! \brief Array index of the blue component value.
    //----------------------------------------
    static constexpr uint8_t blueIndex = 2;

    //========================================
    //! \brief Array index of the alpha component value.
    //----------------------------------------
    static constexpr uint8_t alphaIndex = 3;

public:
    //========================================
    //! \brief Constructor.
    //!
    //! By default all color values are 0, i.e. black.
    //----------------------------------------
    RgbaColor() : m_rgba{{static_cast<ColorValue>(0)}} {}

    //========================================
    //! \brief Constructor.
    //! \param[in] red    Value used for red component.
    //! \param[in] green  Value used for green component.
    //! \param[in] blue   Value used for blue component.
    //! \param[in] alpha  Value used for alpha component.
    //----------------------------------------
    RgbaColor(const ColorValue red, const ColorValue green, const ColorValue blue, const ColorValue alpha)
      : m_rgba{red, green, blue, alpha}
    {}
    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  The other RgbaColor the component values
    //!                   are copied from.
    //----------------------------------------
    RgbaColor(const RgbaColor& other) : m_rgba{other.m_rgba} {}

    //========================================
    //! \brief Move constructor.
    //! \param[in,out] other  The other RgbaColor the component values
    //!                       are moved from.
    //----------------------------------------
    RgbaColor(RgbaColor&& other) : m_rgba{other.m_rgba} {}

    //========================================
    //! \brief Assignment operator
    //! \param[in] other  The other RgbaColor the component values
    //!                   are moved from.
    //----------------------------------------
    RgbaColor& operator=(const RgbaColor& other)
    {
        m_rgba = other.m_rgba;
        return *this;
    }

    //========================================
    //! \brief Move assignment operator
    //! \param[in,out] other  The other RgbaColor the component values
    //!                       are moved from.
    //----------------------------------------
    RgbaColor operator=(RgbaColor&& other)
    {
        m_rgba = other.m_rgba;
        return *this;
    }

public:
    //========================================
    //! \brief Get the red color component value.
    //! \return Return the red color component value.
    //----------------------------------------
    ColorValue red() const { return m_rgba[redIndex]; }

    //========================================
    //! \brief Get the green color component value.
    //! \return Return the green color component value.
    //----------------------------------------
    ColorValue green() const { return m_rgba[greenIndex]; }

    //========================================
    //! \brief Get the blue color component value.
    //! \return Return the blue color component value.
    //----------------------------------------
    ColorValue blue() const { return m_rgba[blueIndex]; }

    //========================================
    //! \brief Get the alpha color component value.
    //! \return Return the alpha color component value.
    //----------------------------------------
    ColorValue alpha() const { return m_rgba[alphaIndex]; }

public:
    //========================================
    //! \brief Get the color values as an array.
    //! \return Return a const reference to the color value array.
    //----------------------------------------
    const ColorValueArray& getColorValueArray() const { return m_rgba; }

    //========================================
    //! \brief Get the color values as an array.
    //! \return Return a const reference to the color value array.
    //----------------------------------------
    ColorValueArray& getColorValueArray() { return m_rgba; }

public:
    //========================================
    //! \brief Set the red color component value.
    //! \param[in] redValue  The new red color component value.
    //----------------------------------------
    void setRed(const ColorValue redValue) { m_rgba[redIndex] = redValue; }

    //========================================
    //! \brief Set the green color component value.
    //! \param[in] greenValue  The new green color component value.
    //----------------------------------------
    void setGreen(const ColorValue greenValue) { m_rgba[greenIndex] = greenValue; }

    //========================================
    //! \brief Set the blue color component value.
    //! \param[in] blueValue  The new blue color component value.
    //----------------------------------------
    void setBlue(const ColorValue blueValue) { m_rgba[blueIndex] = blueValue; }

    //========================================
    //! \brief Set the alpha color component value.
    //! \param[in] alphaValue  The new alpha color component value.
    //----------------------------------------
    void setAlpha(const ColorValue alphaValue) { m_rgba[alphaIndex] = alphaValue; }

public:
    //========================================
    //! \brief Set the color component values as an array.
    //! \param[in] newRgba  The color component values as an array.
    //----------------------------------------
    void setColorValueArray(const ColorValueArray& newRgba) { m_rgba = newRgba; }

    //========================================
    //! \brief Set the color component values as an array.
    //! \param[in,out] newRgba  The color component values as an array.
    //----------------------------------------
    void setColorValueArray(ColorValueArray&& newRgba) { m_rgba = std::move(newRgba); }

private:
    //========================================
    //! \brief The color component values.
    //----------------------------------------
    ColorValueArray m_rgba;
}; // RgbaColor

//==============================================================================

using RgbaColor8     = RgbaColor<uint8_t>;
using RgbaColorFloat = RgbaColor<float>;

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The RgbaColor, that shall be compared.
//! \param[in] rhs  The RgbaColor, that first RgbaColor shall be compared to.
//! \return \c True, if the RgbaColor are identical, \c false otherwise.
//------------------------------------------------------------------------------
template<typename ColorValue>
bool operator==(const RgbaColor<ColorValue>& lhs, const RgbaColor<ColorValue>& rhs)
{
    return lhs.getColorValueArray() == rhs.getColorValueArray();
}

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The RgbaColor, that shall be compared.
//! \param[in] rhs  The RgbaColor, that first RgbaColor shall be compared to.
//! \return \c True, if the RgbaColor are not identical, \c false otherwise.
//------------------------------------------------------------------------------
template<typename ColorValue>
bool operator!=(const RgbaColor<ColorValue>& lhs, const RgbaColor<ColorValue>& rhs)
{
    return lhs.getColorValueArray() != rhs.getColorValueArray();
}

//==============================================================================

template<typename ColorValue>
inline void readBE(std::istream& is, RgbaColor<ColorValue>& color)
{
    auto& arr = color.getColorValueArray();
    for (auto& value : arr)
    {
        readBE(is, value);
    }
}

//==============================================================================

template<typename ColorValue>
inline void writeBE(std::ostream& os, const RgbaColor<ColorValue>& color)
{
    for (const auto value : color.getColorValueArray())
    {
        writeBE(os, value);
    }
}

//==============================================================================

template<typename ColorValue>
inline void readLE(std::istream& is, RgbaColor<ColorValue>& color)
{
    auto& arr = color.getColorValueArray();
    for (auto& value : arr)
    {
        readLE(is, value);
    }
}

//==============================================================================

template<typename ColorValue>
inline void writeLE(std::ostream& os, const RgbaColor<ColorValue>& color)
{
    for (const auto value : color.getColorValueArray())
    {
        writeLE(os, value);
    }
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
