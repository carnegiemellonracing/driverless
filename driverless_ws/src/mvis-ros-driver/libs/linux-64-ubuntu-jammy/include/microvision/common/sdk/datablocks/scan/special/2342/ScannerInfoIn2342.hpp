//=================information==================================================
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
//! \date Jan 18, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/MountingPositionWithDeviation.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/ScannerDirectionListIn2342.hpp>
#include <microvision/common/sdk/datablocks/PerceptionDataInfo.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Scanner information in the format used by a processed low bandwidth MOVIA scan.
//------------------------------------------------------------------------------
class ScannerInfoIn2342 final
{
public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ScannerInfoIn2342() = default;

    //========================================
    //! \brief Default copy-constructor.
    //!
    //! \param[in] src  Object to create a copy from.
    //----------------------------------------
    ScannerInfoIn2342(const ScannerInfoIn2342& src) = default;

    //========================================
    //! \brief Default assignment operator.
    //!
    //! \param[in] src  Object to be assigned
    //! \return A reference to \c this.
    //----------------------------------------
    ScannerInfoIn2342& operator=(const ScannerInfoIn2342& src) = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~ScannerInfoIn2342() = default;

public: // getter
    //========================================
    //! \brief Get the header of this scan.
    //!
    //! \return The header data of this Scan.
    //----------------------------------------
    const PerceptionDataInfo& getDataInfo() const { return m_dataInfo; }

    //========================================
    //! \brief Get the ScannerDirectionList of this scanner.
    //!
    //! \return The direction data of this Scanner.
    //----------------------------------------
    const ScannerDirectionListIn2342& getDirections() const { return m_directionList; }

    //========================================
    //! \brief Get the mountingPosition of this scanner.
    //!
    //! \return The mountingPosition of this Scanner.
    //----------------------------------------
    const MountingPositionWithDeviation<float>& getMountingPosition() const { return m_mountingPosition; }

    //========================================
    //! \brief Get the value for the hash.
    //!
    //! \return The hash value.
    //----------------------------------------
    uint32_t getHashValue() const { return m_hash; }

    //========================================
    //! \brief Get the value for the cyclic redundancy check.
    //!
    //! \return The check value.
    //----------------------------------------
    uint32_t getCyclicRedundancyCheckValue() const { return m_crc; }

public: // setter
    //========================================
    //! \brief Set the data info of this scan.
    //!
    //! \param[in] dataInfo  The new data info.
    //----------------------------------------
    void setDataInfo(const PerceptionDataInfo& dataInfo) { m_dataInfo = dataInfo; }

    //========================================
    //! \brief Set the direction data of this scanner.
    //!
    //! \param[in] directionList  The new direction data.
    //----------------------------------------
    void setDirections(const ScannerDirectionListIn2342& directionList) { m_directionList = directionList; }

    //========================================
    //! \brief Set the mountingPosition of this scanner.
    //!
    //! \param[in] mountingPosition  The new mountingPosition.
    //----------------------------------------
    void setMountingPosition(const MountingPositionWithDeviation<float>& mountingPosition)
    {
        m_mountingPosition = mountingPosition;
    }

    //========================================
    //! \brief Set the value for the hash.
    //!
    //! \param[in] value  The hash value.
    //----------------------------------------
    void setHashValue(const uint32_t value) { m_hash = value; }

    //========================================
    //! \brief Set the value for the cyclic redundancy check.
    //!
    //! \param[in] value  The check value.
    //----------------------------------------
    void setCyclicRedundancyCheckValue(const uint32_t value) { m_crc = value; }

private:
    PerceptionDataInfo m_dataInfo{}; //!< The perception API data header.
    ScannerDirectionListIn2342 m_directionList{}; //!< The directions of each scanner pixel.
    MountingPositionWithDeviation<float> m_mountingPosition{}; //!< The mounting position of the scanner.
    uint32_t m_hash{0}; //!< The hash over directionList and mountingPosition.
    uint32_t m_crc{0}; //!< The cyclic redundancy check value over header, directionList, sensorPose and hash.

}; // ScannerInfoIn2342

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const ScannerInfoIn2342& lhs, const ScannerInfoIn2342& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const ScannerInfoIn2342& lhs, const ScannerInfoIn2342& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
