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
//! \date Jan 22, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/metainformationlist/MetaInformationBase.hpp>
#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationList7110.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Container for a list of meta information datas
//!
//! Special data container: \ref microvision::common::sdk::MetaInformationList7110
//------------------------------------------------------------------------------
class MetaInformationList final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const MetaInformationList& lhs, const MetaInformationList& rhs);

public:
    using MetaInformationSPtrVector = std::vector<MetaInformationBaseSPtr>;
    using MetaInformationMap
        = std::unordered_map<MetaInformationBase::MetaInformationType, MetaInformationSPtrVector, EnumClassHash>;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.metainformationlist"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    MetaInformationList() : DataContainerBase() {}
    ~MetaInformationList() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    const MetaInformationMap& getMetaInformationMap() const { return m_delegate.getMetaInformationMap(); }

    void addMetaInformation(const MetaInformationBaseSPtr info) { m_delegate.addMetaInformation(info); }

    uint32_t getNumberOfMetaInformationElements() const { return m_delegate.getNumberOfMetaInformationElements(); }

    //========================================
    //! \brief How many different meta information types are stored in this container.
    //! \return The count of different meta information types.
    //----------------------------------------
    uint32_t getNumberOfMetaInformationTypes() const { return m_delegate.getNumberOfMetaInformationTypes(); }

    //========================================
    //! \brief Return a vector of meta information stored in this container.
    //! \param[in] infoType  type of meta information wanted
    //! \return The meta information of the given type.
    //----------------------------------------
    template<class T>
    std::vector<std::shared_ptr<T>> getMetaInformations(const MetaInformationBase::MetaInformationType infoType)
    {
        return m_delegate.getMetaInformations<T>(infoType);
    }

    void deleteInformation(const MetaInformationBaseSPtr& info) { return m_delegate.deleteInformation(info); }
    void deleteInformationForType(const MetaInformationBase::MetaInformationType infoType)
    {
        return m_delegate.deleteInformationForType(infoType);
    }

    void printStatistic(std::ostream& os) const { m_delegate.printStatistic(os); }

private:
    MetaInformationList7110 m_delegate;
}; // MetaInformationList

//==============================================================================

inline bool operator==(const MetaInformationList& lhs, const MetaInformationList& rhs)
{
    return lhs.m_delegate == rhs.m_delegate;
}

inline bool operator!=(const MetaInformationList& lhs, const MetaInformationList& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
