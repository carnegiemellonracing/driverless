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
//! \date Jun 5, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/StreamBase.hpp>
#include <microvision/common/sdk/io/DataPackage.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Interface to provide write functionality for data source.
//!
//! Inherit from this interface to provide functionality to write data sources.
//! Aligned to your implementation add an implementation of StreamWriterFactoryExtension.
//!
//! \sa StreamWriterFactory
//------------------------------------------------------------------------------
class DataPackageStreamWriter : public virtual StreamBase
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using StreamBase::open;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~DataPackageStreamWriter() override = default;

public:
    //========================================
    //! \brief Request resource access to append data if possible.
    //! \param[in] append  Either \c true if the resource requested is in append mode, otherwise \c false.
    //----------------------------------------
    virtual bool open(const bool appendMode) { return this->open(nullptr, appendMode); }

public:
    //========================================
    //! \brief Request resource access and move stream.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream) override { return this->open(std::move(ioStream), false); }

public:
    //========================================
    //! \brief Write data package at stream position.
    //! \param[in/out] data  DataPackage to write.
    //! \returns Either \c true if DataPackage successful written, otherwise \c false.
    //----------------------------------------
    virtual bool writePackage(DataPackage& dataPackage) = 0;

    //========================================
    //! \brief Write data package at stream position.
    //! \param[in] data  DataPackage to write.
    //! \returns Either \c true if DataPackage successful written, otherwise \c false.
    //!
    //! \note This method does not change source, index and previous message size in the package header.
    //!       That data is required for serialization but possibly not for your code.
    //----------------------------------------
    virtual bool writePackage(const DataPackage& dataPackage) = 0;

    //========================================
    //! \brief Request resource access to append data if possible.
    //! \param[in] ioStream  Resource stream to move.
    //! \param[in] append    Either \c true if the resource requested is in append mode, otherwise \c false.
    //----------------------------------------
    virtual bool open(IoStreamPtr&& ioStream, const bool appendMode) = 0;
};

//==============================================================================

//========================================
//! \brief Nullable DataPackageStreamWriter pointer.
//----------------------------------------
using DataPackageStreamWriterPtr = std::unique_ptr<DataPackageStreamWriter>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
