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
//! \date Jun 14, 2019
//------------------------------------------------------------------------------
//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

#include <microvision/common/sdk/datablocks/scan/Scan.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2208.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanPointIn2208.hpp>
#include <microvision/common/sdk/datablocks/scan/ScanPoint.hpp>

#include <microvision/common/sdk/datablocks/scan/special/ScalaRayCorrection.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Importer<Scan, DataTypeId::DataType_Scan2208> : public RegisteredImporter<Scan, DataTypeId::DataType_Scan2208>
{
    friend bool operator==(const Scan& lhs, const Scan2208& rhs);

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
    //! map special 2208 scan point flags to general scan point flags
    static uint16_t convertFlags2208(const uint16_t flags);

    static float getVerticalAngle(const uint8_t layer, const float vBeamDivergence);

    //! convert polar coordinate 2208 scan point to cartesian general scan point
    //! for some devices it is required to do the scala ray angle correction
    static void convertScanPoint2208(const ScanPointIn2208& scanPoint2208,
                                     ScanPoint& scanPoint,
                                     const bool isRearMirrorSide,
                                     float degreesPerTick,
                                     bool doScalaRayAngleCorrection);
}; // ScanImporter2208

//==============================================================================

using ScanImporter2208 = Importer<Scan, DataTypeId::DataType_Scan2208>;

bool operator==(const Scan& lhs, const Scan2208& rhs);

inline bool operator!=(const Scan& lhs, const Scan2208& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
