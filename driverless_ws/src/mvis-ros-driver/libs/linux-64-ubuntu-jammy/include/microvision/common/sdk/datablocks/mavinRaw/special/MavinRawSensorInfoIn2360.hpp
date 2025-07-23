//==============================================================================
//! \file
//!
//! \brief Data type to store mavin raw frame format extracted from header data.
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

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/MountingPosition.hpp>
#include <microvision/common/sdk/datablocks/mavinRaw/MavinDataFormat.hpp>

#include <microvision/common/sdk/io.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Ldmi static info helper for reassembling ldmi raw frame package.
//------------------------------------------------------------------------------
class MavinRawSensorInfoIn2360 final
{
public:
    //========================================
    //! \brief Get size of binary format.
    //! \return Size of binary format.
    //----------------------------------------
    static constexpr std::streamsize serializedSize()
    {
        return sizeof(uint8_t) // device id
               + sizeof(uint8_t) // data format
               + (sizeof(float) * 6); // mounting position
    }

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    MavinRawSensorInfoIn2360();

    //========================================
    //! \brief Move constructor.
    //! \param[in] other  Another instance of frame static info.
    //----------------------------------------
    MavinRawSensorInfoIn2360(MavinRawSensorInfoIn2360&& other);

    //========================================
    //! \brief Copy constructor.
    //! \param[in] other  Another instance of frame static info.
    //----------------------------------------
    MavinRawSensorInfoIn2360(const MavinRawSensorInfoIn2360& other);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~MavinRawSensorInfoIn2360() = default;

public:
    //========================================
    //! \brief Move assignment operator.
    //! \param[in] other  Another instance of frame static info.
    //! \return Reference of this.
    //----------------------------------------
    MavinRawSensorInfoIn2360& operator=(MavinRawSensorInfoIn2360&& other);

    //========================================
    //! \brief Copy assignment operator.
    //! \param[in] other  Another instance of frame static info.
    //! \return Reference of this.
    //----------------------------------------
    MavinRawSensorInfoIn2360& operator=(const MavinRawSensorInfoIn2360& other);

public:
    //========================================
    //! \brief Compares two ldmi raw static infos for equality.
    //! \param[in] lhs  Ldmi raw static info in 2353.
    //! \param[in] rhs  Ldmi raw static info in 2353.
    //! \return Either \c true if both static infos are equal or otherwise \c
    //! false.
    //----------------------------------------
    friend bool operator==(const MavinRawSensorInfoIn2360& lhs, const MavinRawSensorInfoIn2360& rhs);

    //========================================
    //! \brief Compares two ldmi raw static infos for inequality.
    //! \param[in] lhs  Ldmi raw static info in 2353.
    //! \param[in] rhs  Ldmi raw static info in 2353.
    //! \return Either \c true if both static infos are unequal or otherwise \c
    //! false.
    //----------------------------------------
    friend bool operator!=(const MavinRawSensorInfoIn2360& lhs, const MavinRawSensorInfoIn2360& rhs);

public: // getter
    //========================================
    //! \brief Get setup device id.
    //! \return Setup device id.
    //----------------------------------------
    uint8_t getDeviceId() const;

    //========================================
    //! \brief Get expected data format.
    //! \return Expected data format.
    //----------------------------------------
    MavinDataFormat getDataFormat() const;

    //========================================
    //! \brief Get device mounting position.
    //! \return Mounting position of device.
    //----------------------------------------
    const MountingPosition<float>& getMountingPosition() const;

public: // setter
    //========================================
    //! \brief Set setup device id.
    //! \param[in] deviceId  Setup device id.
    //----------------------------------------
    void setDeviceId(const uint8_t deviceId);

    //========================================
    //! \brief Set expected data format.
    //! \param[in] dataFormat  Expected data format.
    //----------------------------------------
    void setDataFormat(const MavinDataFormat dataFormat);

    //========================================
    //! \brief Set mounting position of device.
    //! \param[in] mountingPosition  Mounting position of device.
    //----------------------------------------
    void setMountingPosition(const MountingPosition<float>& mountingPosition);

private:
    //========================================
    //! \brief Setup device id.
    //----------------------------------------
    uint8_t m_deviceId;

    //========================================
    //! \brief Expected data format.
    //----------------------------------------
    MavinDataFormat m_dataFormat;

    //========================================
    //! \brief Device mounting position.
    //----------------------------------------
    MountingPosition<float> m_mountingPosition;

}; // MavinRawSensorInfoIn2360

//==============================================================================
//! \brief Read MavinRawSensorInfoIn2360 from binary data stream.
//! \param[in, out] is      Binary data stream.
//! \param[out]     value   Instance of MavinRawSensorInfoIn2360 to be filled.
//------------------------------------------------------------------------------
template<>
inline void readLE<MavinRawSensorInfoIn2360>(std::istream& is, MavinRawSensorInfoIn2360& value)
{
    uint8_t deviceId{};
    uint8_t dataFormat{};
    MountingPosition<float> mountingPosition{};

    readLE(is, deviceId);
    readLE(is, dataFormat);
    readLE(is, mountingPosition);

    value.setDeviceId(deviceId);
    value.setDataFormat(static_cast<MavinDataFormat>(dataFormat));
    value.setMountingPosition(mountingPosition);
}

//==============================================================================
//! \brief Write MavinRawSensorInfoIn2360 to binary data stream.
//! \param[in, out] os      Binary data stream.
//! \param[out]     value   Instance of MavinRawSensorInfoIn2360 to write.
//------------------------------------------------------------------------------
template<>
inline void writeLE<MavinRawSensorInfoIn2360>(std::ostream& os, const MavinRawSensorInfoIn2360& value)
{
    writeLE(os, value.getDeviceId());
    writeLE(os, static_cast<uint8_t>(value.getDataFormat()));
    writeLE(os, value.getMountingPosition());
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
