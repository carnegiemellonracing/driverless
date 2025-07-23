//==============================================================================
//! \file
//!
//! \brief Configuration factory extension for the prototype MOVIA B0 sensor device.
//!
//! \note Please note that using recent MOVIA sensors require the movia-device-plugin to be loaded!
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 04, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/extension/ConfigurationFactoryExtension.hpp>

#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief SDK specific configuration factory extension.
//!
//! This class provides the functionality to create SDK specific configurations.
//!
//! \note Please note that using some MOVIA sensors require the movia-device-plugin to be loaded!
//!
//! \note Choose the default parameter set for your MOVIA B0 device (configuration type: "moviab0") by sensor release version/name.
//! Default parameter set   | Description
//! ----------------------- | ---------------------------------------------------------------------------------------------
//! INLSB-CPA-0.2.8         | Defines the default value for the transport protocol chain as MoviaTransportProtocol::Iutp
//! INLSB-CPA-0.2.12        | Defines the default value for the transport protocol chain as MoviaTransportProtocol::Iutp
//! INLSB-CPA-0.2.16        | Defines the default value for the transport protocol chain as MoviaTransportProtocol::Iutp
//! INLSB-CPA-0.2.17        | Defines the default value for the transport protocol chain as MoviaTransportProtocol::Intp
//! INLSB-CPA-0.4.2         | Defines the default value for the transport protocol chain as MoviaTransportProtocol::Intp
//! L                       | Defines the default value for the transport protocol chain as MoviaTransportProtocol::Iutp
//------------------------------------------------------------------------------
class SdkConfigurationFactoryExtension : public ConfigurationFactoryExtension
{
public:
    //========================================
    //! \brief MOVIA default parameter set for 'INLSB-CPA-0.2.8'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string moviaLidarB028;

    //========================================
    //! \brief MOVIA default parameter set for 'INLSB-CPA-0.2.12'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string moviaLidarB0212;

    //========================================
    //! \brief MOVIA default parameter set for 'INLSB-CPA-0.2.16'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string moviaLidarB0216;

    //========================================
    //! \brief MOVIA default parameter set for 'INLSB-CPA-0.2.17'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string moviaLidarB0217;

    //========================================
    //! \brief MOVIA default parameter set for 'INLSB-CPA-0.4.2'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string moviaLidarB042;

    //========================================
    //! \brief MOVIA default parameter set for 'L'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string moviaLidarL;

    //========================================
    //! \brief MOVIA default parameter set ids.
    //----------------------------------------
    static MICROVISION_SDK_API const std::vector<std::string> moviaDefaultParameterSetIds;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~SdkConfigurationFactoryExtension() override;

public:
    //========================================
    //! \brief Get list of default parameter set ids.
    //! \param[in] configurationType  Configuration type.
    //! \return List with all possible default parameter set ids.
    //----------------------------------------
    const std::vector<std::string>& getDefaultParameterSets(const std::string& configurationType) const override;

    //========================================
    //! \brief Create a configuration by configuration type.
    //! \param[in] configurationType    Unique human readable configuration type name string of the wanted configuration.
    //! \param[in] defaultParameterSet  Human readable identifier of default parameter set,
    //!                                 which will pass the default values by configuration constructor.
    //! \return Either a shared pointer to an instance of the \c Configuration or otherwise \c nullptr.
    //----------------------------------------
    ConfigurationPtr createConfiguration(const std::string& configurationType,
                                         const std::string& defaultParameterSet) const override;

}; // class ConfigurationFactoryExtension

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
