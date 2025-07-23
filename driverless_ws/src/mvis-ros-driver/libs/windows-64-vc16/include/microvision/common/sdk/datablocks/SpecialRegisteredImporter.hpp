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
//! \date Feb 13, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/SpecialImporterBase.hpp>

#include <microvision/common/sdk/listener/DataContainerListener.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================
//!\brief Template class whose specializations will be derived from
//!       SpecialRegisteredImporter.
//!\date Feb 13, 2018
//------------------------------------------------------------------------------
template<class ContainerType, DataTypeId::DataType id, class SpecialContainer>
class SpecialImporter
{
};

//==============================================================================
//!\brief Intermediate class between SpecialImporterBase and SpecialImporter which
//!       provides registration to devices.
//!\date Jan 9, 2018
//------------------------------------------------------------------------------
template<class ContainerType, DataTypeId::DataType dataType, class SpecialContainer>
class SpecialRegisteredImporter : public SpecialImporterBase<typename SpecialContainer::CommonBase>
{
public:
    static SpecialImporterBase<typename SpecialContainer::CommonBase>* create()
    {
        return new SpecialImporter<ContainerType, dataType, SpecialContainer>;
    }
    static std::shared_ptr<SpecialContainer> createContainerStatic() { return std::make_shared<SpecialContainer>(); }
    static DataTypeId getDataTypeStatic() { return static_cast<DataTypeId>(dataType); }

public:
    SpecialRegisteredImporter() : SpecialImporterBase<typename SpecialContainer::CommonBase>() {}
    SpecialRegisteredImporter(const SpecialRegisteredImporter&) = delete;
    SpecialRegisteredImporter& operator=(const SpecialRegisteredImporter&) = delete;

    ~SpecialRegisteredImporter() override = default;

public:
    DataTypeId getDataType() const final { return getDataTypeStatic(); }

    std::shared_ptr<typename SpecialContainer::CommonBase> createContainer() const final
    {
        return createContainerStatic();
    }

public:
    bool callListener(DataContainerListenerBase* l, const typename SpecialContainer::CommonBase& s) const override
    {
        if (auto* lImpl = dynamic_cast<DataContainerSpecialListener<ContainerType, dataType, SpecialContainer>*>(l))
        {
            lImpl->onData(dynamic_cast<const SpecialContainer*>(&s));
            return true;
        }
        else
        {
            LOGTRACE(this->m_logger, "Dynamic cast failed");
            return false;
        }
    }

private:
    static const typename SpecialImporterBase<typename SpecialContainer::CommonBase>::ImporterRegisterId
        registeredImporterInitial;
    static const typename SpecialImporterBase<typename SpecialContainer::CommonBase>::ImporterRegisterId
        registeredImporter;
}; // class SpecialRegisteredImporter

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
