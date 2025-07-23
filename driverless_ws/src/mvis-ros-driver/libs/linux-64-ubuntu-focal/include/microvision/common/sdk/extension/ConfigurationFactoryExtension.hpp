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
//! \date Feb 04, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/Configuration.hpp>

#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface to extend configuration factory.
//!
//! This interface will provide functionality to create configurations.
//! A derived extension is then registered at the factory to enable creation of configurations of the type the extension provides.
//------------------------------------------------------------------------------
class ConfigurationFactoryExtension
{
public:
    //========================================
    //! \brief Empty list of default parameter set ids.
    //----------------------------------------
    static const std::vector<std::string> emptyListOfDefaultParameterSetIds;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~ConfigurationFactoryExtension() = default;

public:
    //========================================
    //! \brief Get list of default parameter set ids.
    //! \param[in] configurationType  Configuration type.
    //! \return List with all possible default parameter set ids.
    //----------------------------------------
    virtual const std::vector<std::string>& getDefaultParameterSets(const std::string& configurationType) const;

    //========================================
    //! \brief Create a configuration by configuration type.
    //! \param[in] configurationType    Unique human readable configuration type name string of the wanted configuration.
    //! \param[in] defaultParameterSet  Human readable identifier of default parameter set,
    //!                                 which will pass the default values by configuration constructor.
    //! \return Either a shared pointer to an instance of the \c Configuration or otherwise \c nullptr.
    //----------------------------------------
    virtual ConfigurationPtr createConfiguration(const std::string& configurationType,
                                                 const std::string& defaultParameterSet) const = 0;
}; // class ConfigurationFactoryExtension

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
