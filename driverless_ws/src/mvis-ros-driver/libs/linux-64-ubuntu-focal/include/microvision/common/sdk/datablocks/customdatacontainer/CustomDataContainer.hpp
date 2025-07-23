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
//! \date Aug 16th, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>
#include <microvision/common/sdk/MsgBuffer.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <sstream>
#include <random>
#include <string>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief idc custom datatype
//!
//! Contains a uuid/GUID and a buffer of data.
//------------------------------------------------------------------------------
class CustomDataContainer : public DataContainerBase, public CustomDataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const CustomDataContainer& lhs, const CustomDataContainer& rhs);

public:
    //========================================
    //! \brief Empty constructor calling CustomDataContainerBase constructor.
    //----------------------------------------
    CustomDataContainer();

    //========================================
    //! \brief Default destructor
    //----------------------------------------
    ~CustomDataContainer() override = default;

public:
    //========================================
    //! \brief Get the static hash value of the class id.
    //!
    //! \return The hash value specifying the custom data container class.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

    //========================================
    //! \brief Get the unique id of this custom data type.
    //!
    //! \return The uuid/GUID specifying this custom data container.
    //----------------------------------------
    std::string getUuid() const override { return m_uuid; }

    //========================================
    //! \brief Get the name of this custom data type.
    //!
    //! \return The human readable name of this custom data container.
    //----------------------------------------
    std::string getName() const override { return m_customName; }

    //========================================
    //! \brief Get a description of this custom data type.
    //!
    //! \return The human readable description of the use of this custom data container.
    //----------------------------------------
    std::string getDescription() const override { return m_customDescription; }

public: // getter
    //========================================
    //! \brief Get size of 'unknown' content of this custom data container.
    //!
    //! \return Content size in bytes.
    //----------------------------------------
    uint32_t getContentSize() const override { return m_data ? static_cast<uint32_t>(m_data.get()->size()) : 0; }

    //========================================
    //! \brief Get the 'unknown' content of this custom data container as a char vector.
    //!
    //! \return Pointer to char vector containing the content.
    MsgBufferBase::ConstCharVectorPtr getContent() const { return m_data; }

public: // setter
    //========================================
    //! \brief Set the unique id of this custom data type.
    //!
    //! Note: Setting the wrong id will lead to misinterpretation of the data when exported and imported again!
    //!
    //! \param[in] uuid  New uuid/GUID of the custom data container.
    //----------------------------------------
    void setUuid(const std::string& uuid) { m_uuid = uuid; }

    //========================================
    //! \brief Set the name of this custom data type.
    //!
    //! \param[in] uuid  New name for the custom data container.
    //----------------------------------------
    void setName(const std::string& name) { m_customName = name; }

    //========================================
    //! \brief Set the description of this custom data type.
    //!
    //! \param[in] uuid  New description for the custom data container.
    //----------------------------------------
    void setDescription(const std::string& desc) { m_customDescription = desc; }

    //========================================
    //! \brief Set the binary content of this custom data type.
    //!
    //! Note: Setting the wrong content for the uuid will lead to misinterpretation of the data when exported and imported again!
    //!
    //! \param[in] data  New data in the custom data container.
    //----------------------------------------
    void setContent(const MsgBufferBase::ConstCharVectorPtr& data) { m_data = data; }

protected:
    MsgBufferBase::ConstCharVectorPtr m_data; // byte content data

    std::string m_uuid{""}; // not static for the general custom data container
    std::string m_customName{""}; // not static for the general custom data container
    std::string m_customDescription{""}; // not static for the general custom data container
}; // CustomDataContainer

//==============================================================================

//========================================
//! \brief Compare two custom data containers for equality.
//!
//! \param[in] lhs  First custom data container to compare.
//! \param[in] rhs  Second to compare.
//! \return \c True if equal, \c false if not.
//!
//! \note Since this is the general custom data container only
//! the size and the binary byte content are compared - this
//! might not be real equality for some types!
//----------------------------------------
bool operator==(const CustomDataContainer& lhs, const CustomDataContainer& rhs);

//========================================
//! \brief Compare two custom data containers for inequality.
//!
//! \param[in] lhs  First custom data container to compare.
//! \param[in] rhs  Second to compare.
//! \return \c True if different, \c false if not.
//!
//! \note Since this is the general custom data container only
//! the size and the binary byte content are compared - this
//! might not be real difference for some types!
//----------------------------------------
bool operator!=(const CustomDataContainer& lhs, const CustomDataContainer& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
