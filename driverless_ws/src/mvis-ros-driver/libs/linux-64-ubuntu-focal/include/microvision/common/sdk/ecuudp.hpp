//==============================================================================
//! \file
//!
//! \brief Includes for MVIS udp ecu device usage.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 18th, 2023
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/extension/DeviceFactory.hpp>
#include <microvision/common/sdk/extension/ConfigurationFactory.hpp>

#include <microvision/common/sdk/config/devices/MvisEcuUdpConfiguration.hpp>
#include <microvision/common/sdk/devices/MvisEcuUdpDevice.hpp>

#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/Scan2340Exporter2340.hpp>

#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342Importer2342.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342Exporter2342.hpp>

#include <microvision/common/sdk/datablocks/scan/special/Scan2209.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2209Importer2209.hpp>
#include <microvision/common/sdk/datablocks/scan/special/Scan2209Exporter2209.hpp>

#include <microvision/common/sdk/datablocks/image/special/Image2404.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2404Importer2404.hpp>
#include <microvision/common/sdk/datablocks/image/special/Image2404Exporter2404.hpp>

#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainer.hpp>
#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainerImporter.hpp>
#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainerExporter.hpp>

#include <microvision/common/sdk/listener/IdcDataPackageListener.hpp>
#include <microvision/common/sdk/listener/DataContainerListener.hpp>

#include <microvision/common/sdk/MicroVisionSdk.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
