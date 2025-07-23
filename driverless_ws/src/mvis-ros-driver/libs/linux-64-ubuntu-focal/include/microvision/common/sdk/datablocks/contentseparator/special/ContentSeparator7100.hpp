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
//! \date Jan 16, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/contentseparator/special/ContentSeparatorTypeIn7100.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Idc content separator
//!
//! General data type: microvision::common::sdk::ContentSeparator
//------------------------------------------------------------------------------
class ContentSeparator7100 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.contentseparator7100"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ContentSeparator7100()           = default;
    ~ContentSeparator7100() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    ContentSeparatorTypeIn7100 getSeparatorType() const { return m_separatorType; }
    uint32_t getContentSize() const { return m_sizeOfContent; }

public: // setter
    void setSeparatorType(const ContentSeparatorTypeIn7100 newSeparatorType) { m_separatorType = newSeparatorType; }
    void setContentSize(const uint32_t size) { m_sizeOfContent = size; }

private:
    ContentSeparatorTypeIn7100 m_separatorType{ContentSeparatorTypeIn7100::Undefined};
    uint32_t m_sizeOfContent{0};
}; // ContentSeparator7100

//==============================================================================

template<>
inline void writeBE<ContentSeparatorTypeIn7100>(std::ostream& os, const ContentSeparatorTypeIn7100& cst)
{
    writeBE(os, uint16_t(cst));
}

//==============================================================================

template<>
inline void readBE<ContentSeparatorTypeIn7100>(std::istream& is, ContentSeparatorTypeIn7100& cst)
{
    uint16_t rd16;
    readBE(is, rd16);
    cst = ContentSeparatorTypeIn7100(rd16);
}

//==============================================================================

//==============================================================================

bool operator==(const ContentSeparator7100& lhs, const ContentSeparator7100& rhs);
bool operator!=(const ContentSeparator7100& lhs, const ContentSeparator7100& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
