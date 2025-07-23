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
//! \date Mar 1, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/SpecialExporterBase.hpp>
#include <microvision/common/sdk/datablocks/commands/Command2010.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<typename SpecialContainerType>
class EmptyCommandExporter : public SpecialExporterBase<CommandCBase>
{
public:
    static constexpr int commandSize{4};

public:
    EmptyCommandExporter() : SpecialExporterBase<CommandCBase>() {}
    virtual ~EmptyCommandExporter() {}

public:
    //========================================
    //!\brief Get the DataType of exporter/importer.
    //!\return The DataTypeId of the data this exporter/importer
    //!        can handle.
    //----------------------------------------
    virtual CommandCBase::KeyType getSpecialType() const override { return SpecialContainerType::key; }

public:
    //========================================
    //!\brief Get serializable size of data from exporter/importer.
    //!\return Number of Bytes used by data type.
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const CommandCBase& c) const override;

    virtual bool serialize(std::ostream& outStream, const CommandCBase& c) const override;
}; // EmptyCommandExporter

//==============================================================================

template<typename SpecialContainerType>
std::streamsize EmptyCommandExporter<SpecialContainerType>::getSerializedSize(const CommandCBase& s) const
{
    try
    {
        (void)dynamic_cast<const SpecialContainerType&>(s);
    }
    catch (const std::bad_cast&)
    {
        throw ContainerMismatch();
    }

    return static_cast<std::streamsize>(commandSize);
}

//==============================================================================

template<typename SpecialContainerType>
bool EmptyCommandExporter<SpecialContainerType>::serialize(std::ostream& os, const CommandCBase& s) const
{
    const SpecialContainerType* specialType{nullptr};

    try
    {
        specialType = &dynamic_cast<const SpecialContainerType&>(s);
    }
    catch (const std::bad_cast&)
    {
        throw ContainerMismatch();
    }

    const int64_t startPos = streamposToInt64(os.tellp());

    writeLE(os, specialType->m_commandId);
    writeLE(os, specialType->m_reserved);

    return !os.fail() && ((streamposToInt64(os.tellp()) - startPos) == this->getSerializedSize(*specialType));
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
