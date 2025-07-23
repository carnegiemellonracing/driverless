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
//! \date May 24, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/idctrailer/special/IdcTrailer6120.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief IDC Trailer:
//! Data type marking the end of each IDC-file. Does not contain any data.
//!
//! Each IDC-file shall conclude with the IDC-Trailer. The IDC-Trailer does not contain any data,
//! but marks the end of the IDC-File. So it includes simply the header with the data type number.
//------------------------------------------------------------------------------
class IdcTrailer final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.idctrailer"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    IdcTrailer();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~IdcTrailer() = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

protected:
    IdcTrailer6120 m_delegate; // only possible specialization currently
}; // IdcTrailerContainer

//==============================================================================

inline bool operator==(const IdcTrailer&, const IdcTrailer&) { return true; } //< class is empty, i.e. always equal

inline bool operator!=(const IdcTrailer&, const IdcTrailer&) { return false; } //< class is empty, i.e. never inequal

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
