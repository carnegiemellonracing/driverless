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
//! \date Feb 02, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationUnsupportedIn7110.hpp>
#include <microvision/common/sdk/datablocks/metainformationlist/MetaInformationFactory.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <unordered_map>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Idc meta information list
//!
//! The meta information datatype is used to store configurations, version numbers, keywords and other meta information.
//!
//! General data container: \ref microvision::common::sdk::MetaInformationList
//------------------------------------------------------------------------------
class MetaInformationList7110 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;

    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using MetaInformationSPtrVector = std::vector<MetaInformationBaseIn7110SPtr>;
    using MetaInformationMap
        = std::unordered_map<MetaInformationBaseIn7110::MetaInformationType, MetaInformationSPtrVector, EnumClassHash>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.metainformationlist7110"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    MetaInformationList7110() : SpecializedDataContainer(), m_metaInformation() {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~MetaInformationList7110() = default;

public:
    //========================================
    //! \brief Get the static hash value of the class id.
    //!
    //! \return The hash value specifying the custom data container class.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Get the map with all meta information in this list.
    //!
    //! \return The map with all meta information in this list.
    //----------------------------------------
    const MetaInformationMap& getMetaInformationMap() const { return m_metaInformation; }

    //========================================
    //! \brief Add a meta information element to this list.
    //!
    //! \param[in] info  The meta information element to add.
    //----------------------------------------
    void addMetaInformation(const MetaInformationBaseIn7110SPtr info);

    //========================================
    //! \brief Get the total number of meta information elements in this list.
    //!
    //! \return The total number of meta information elements in this list.
    //----------------------------------------
    uint32_t getNumberOfMetaInformationElements() const;

    //========================================
    //! \brief Get the number of different meta information types in this list.
    //!
    //! \return The number of different meta information types in this list.
    //----------------------------------------
    uint32_t getNumberOfMetaInformationTypes() const;

    //========================================
    //! \brief Get all meta information entries contained in this list for a given type.
    //!
    //! \tparam    T         Type of meta information in the result vector.
    //! \param[in] infoType  Type of meta information to get.
    //! \return  A vector with all meta information of the given type.
    //----------------------------------------
    template<class T>
    std::vector<std::shared_ptr<T>>
    getMetaInformations(const MetaInformationBaseIn7110::MetaInformationType infoType) const
    {
        std::vector<std::shared_ptr<T>> ret;

        MetaInformationMap::const_iterator metaInformationsIter = m_metaInformation.find(infoType);
        if (metaInformationsIter == m_metaInformation.end())
        {
            return ret;
        }

        for (const MetaInformationBaseIn7110SPtr& info : metaInformationsIter->second)
        {
            ret.push_back(std::dynamic_pointer_cast<T>(info));
        }
        return ret;
    }

    //========================================
    //! \brief Add or replace the vector with meta information for a given type.
    //!
    //! \tparam    T                 Type of meta information in the parameter vector.
    //! \param[in] infoType          Type of meta information to set.
    //! \param[in] metainformation  The meta information to be added or replaced.
    //----------------------------------------
    template<class T>
    void setMetaInformations(const MetaInformationBaseIn7110::MetaInformationType infoType,
                             const std::vector<std::shared_ptr<T>>& metaInformations)
    {
        // Try to insert an empty meta information vector for the given type.
        std::pair<MetaInformationMap::iterator, bool> insertResult
            = m_metaInformation.insert(std::make_pair(infoType, std::vector<MetaInformationBaseIn7110SPtr>()));
        if (insertResult.second == false)
        {
            // Insertion failed, because there is an element with this key in the list already. The iterator
            // (insertResult.first) points to this element -> clear its meta information vector.
            insertResult.first->second.clear();
        }
        // else: insertion was successful, the iterator (insertResult.first) points to the inserted element.

        // Add / replace meta information.
        for (const std::shared_ptr<T>& info : metaInformations)
        {
            insertResult.first->second.push_back(info);
        }
    }

    //========================================
    //! \brief Remove a single meta information from the list.
    //!
    //! \param[in] info  Meta information to be removed.
    //----------------------------------------
    void deleteInformation(const MetaInformationBaseIn7110SPtr& info);

    //========================================
    //! \brief Remove all meta information of the given type from the list.
    //!
    //! \param[in] infoType  Type of meta information to be removed.
    //----------------------------------------
    void deleteInformationForType(const MetaInformationBaseIn7110::MetaInformationType infoType);

    //========================================
    //! \brief Print a statistic about this list to the given stream.
    //!
    //! \param[in, out] os  Stream to write the statistic to.
    //----------------------------------------
    void printStatistic(std::ostream& os) const;

protected:
    //========================================
    //! \brief Common header for all meta information.
    //----------------------------------------
    class MetaInformationHeader
    {
    public:
        //========================================
        //! \brief Read data from the given stream and fill this meta information header (deserialization).
        //!
        //! \param[in, out] is      Input data stream
        //! \return \c True if deserialization succeeds, \c false otherwise.
        //----------------------------------------
        bool deserialize(std::istream& is);

    public:
        MetaInformationBaseIn7110::MetaInformationType m_type;
        NtpTime m_timeStamp;
        uint32_t m_payloadSize;
    }; // MetaInformationHeader

protected:
    //========================================
    //! \brief Check whether two meta information objects are the same.
    //!
    //! \param[in] info1  left object to compare
    //! \param[in] info2  right object to compare
    //! \return \c True if the objects are the same, \c false otherwise.
    //----------------------------------------
    static bool isSame(const MetaInformationBaseIn7110SPtr info1, const MetaInformationBaseIn7110SPtr info2)
    {
        return info1 == info2;
    }

protected:
    //MetaInformationMap m_metaInformations; //!< Deprecated: replaced by m_metaInformation
    MetaInformationMap m_metaInformation; //!< Map of meta information.

}; // MetaInformationList

//==============================================================================

//==============================================================================
//! \brief Test meta information list objects for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const MetaInformationList7110& lhs, const MetaInformationList7110& rhs);

//==============================================================================
//! \brief Test meta information list objects for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const MetaInformationList7110& lhs, const MetaInformationList7110& rhs) { return !(lhs == rhs); }

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
