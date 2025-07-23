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
//! \date March 23, 2018
//------------------------------------------------------------------------------
//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

#include <microvision/common/sdk/datablocks/scan/Scan.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2202.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanPointIn2202.hpp>
#include <microvision/common/sdk/datablocks/scan/ScanPoint.hpp>

#include <microvision/common/sdk/datablocks/scan/special/ScalaRayCorrection.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Importer<Scan, DataTypeId::DataType_Scan2202> : public RegisteredImporter<Scan, DataTypeId::DataType_Scan2202>
{
    friend bool operator==(const Scan& lhs, const Scan2202& rhs);

public:
    //========================================
    //! \brief Empty constructor calling base.
    //----------------------------------------
    Importer() : RegisteredImporter() {}

    //========================================
    //! \brief Copy construction is forbidden.
    //----------------------------------------
    Importer(const Importer&) = delete;

    //========================================
    //! \brief Assignment construction is forbidden.
    //----------------------------------------
    Importer& operator=(const Importer&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Importer() override = default;

public: // implements ImporterBase
    //=================================================
    //! \brief Get the size in bytes that the object occupies when being serialized.
    //!
    //! \param[in] dataContainer  Object to get the serialized size for.
    //! \param[in] configuration  (Optional) Configuration context for import. Default set as nullptr.
    //! \return  the number of bytes used for serialization.
    //-------------------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& dataContainer,
                                      const ConfigurationPtr& configuration = nullptr) const override;

    //=================================================
    //! \brief Read data from the given stream and fill the given data container (deserialization).
    //!
    //! \param[in, out] inputStream     Input data stream
    //! \param[out]     dataContainer   Output container defining the target type (might include conversion).
    //! \param[in]      dataHeader      Metadata prepended to each idc data block.
    //! \param[in]      configuration   (Optional) Configuration context for import. Default set as nullptr.
    //! \return \c True if deserialization succeeds, \c false otherwise.
    //!
    //! \note This method has to be called from outside for deserialization.
    //-------------------------------------------------
    bool deserialize(std::istream& inputStream,
                     DataContainerBase& dataContainer,
                     const IdcDataHeader& dataHeader,
                     const ConfigurationPtr& configuration = nullptr) const override;

private:
    //! map special 2202 scan point flags to general scan point flags
    static uint16_t convertFlags2202(const uint8_t flags);

    static float getVerticalAngle(const uint8_t layer, const float vBeamDivergence);

    //! convert polar coordinate 2202 scan point to cartesian general scan point
    //! for some devices it might be required to do the scala ray angle correction
    static void convertScanPoint2202(const ScanPointIn2202& scanPoint2202,
                                     ScanPoint& scanPoint,
                                     const bool isRearMirrorSide,
                                     const float ticksPerDegree,
                                     const uint32_t scanTimeOffset,
                                     bool doScalaRayAngleCorrection);
}; // ScanImporter2202

//==============================================================================

using ScanImporter2202 = Importer<Scan, DataTypeId::DataType_Scan2202>;

bool operator==(const Scan& lhs, const Scan2202& rhs);

inline bool operator!=(const Scan& lhs, const Scan2202& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
