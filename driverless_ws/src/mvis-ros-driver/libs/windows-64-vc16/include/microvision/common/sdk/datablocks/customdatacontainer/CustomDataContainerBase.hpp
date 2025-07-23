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
//! \date Aug 26th, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <sstream>
#include <random>
#include <string>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief The idc custom datatype base class.
//!
//! This contains a uuid/GUID and content length.
//! Use as base class for all custom data types
//!
//! \note ALL custom data containers have the same datatype and class hash id! They differ in their unique id.
//------------------------------------------------------------------------------
// TODO: future extension or further special base class - general key/value getters interface (KeyValueDataContainerBase)
class CustomDataContainerBase
{
    // every custom data container has the same class id (hence containerType in base). They only differ in uuid/GUID.
    constexpr static const char* const commonCustomContainerType{"sdk.customdatacontainer"};

public:
    //========================================
    //! \brief Get the static hash value of the class id.
    //!
    //! \return The hash value specifying the custom data container class.
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(commonCustomContainerType); }

public:
    //========================================
    //! \brief Create a new random globally unique id (uuid).
    //!
    //! \return uuid/guid as a string.
    //!
    //! Helper function to hide boost uuid.
    //----------------------------------------
    static std::string createUuid()
    {
        boost::uuids::random_generator gen;
        return to_string(gen());
    }

    //========================================
    //! \brief Create a new globally unique id (uuid) from a string.
    //!
    //! String format like: 00000000-0000-0000-0000-000000000000
    //!
    //! \param[in] uuidStr  String in standard uuid-string format.
    //! \return uuid/guid in uuid format.
    //!
    //! Helper function to hide boost uuid.
    //----------------------------------------
    static DataContainerBase::Uuid createUuid(const std::string& uuidStr)
    {
        return boost::uuids::string_generator()(uuidStr);
    };

public: // getter
    //========================================
    //! \brief Get the unique id of this custom data type.
    //!
    //! \return The uuid/GUID specifying this custom data container.
    //----------------------------------------
    virtual std::string getUuid() const = 0;

    //========================================
    //! \brief Get the name of this custom data type.
    //!
    //! \return The human readable name of this custom data container.
    //----------------------------------------
    virtual std::string getName() const = 0;

    //========================================
    //! \brief Get a description of this custom data type.
    //!
    //! \return The human readable description of the use of this custom data container.
    //----------------------------------------
    virtual std::string getDescription() const = 0;

    //========================================
    //! \brief Get size of 'unknown' content of this custom data container.
    //!
    //! \return Content size in bytes.
    //----------------------------------------
    virtual uint32_t getContentSize() const = 0;
}; // CustomDataContainerBase

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
