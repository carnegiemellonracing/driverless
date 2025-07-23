//==============================================================================
//! \file
//!
//! \brief Provides functions to write forward in IdcDataPackage stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 11, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/DataPackageStreamWriter.hpp>
#include <microvision/common/sdk/io/IdcStreamBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Provides functions to write forward in IdcDataPackage stream.
//! \extends IdcStreamBase
//! \extends DataPackageStreamWriter
//------------------------------------------------------------------------------
class IdcDataPackageStreamWriter : public IdcStreamBase, public virtual DataPackageStreamWriter
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using DataPackageStreamWriter::open;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcDataPackageStreamWriter() override = default;

public:
    //========================================
    //! \brief Request resource access to append data if possible and takeover of the stream ownership.
    //! \note Supports only the uri protocols UriProtocol::File and UriProtocol::NoAddr.
    //! \param[in] ioStream     Resource stream handle.
    //! \param[in] append       \c true if the resource requested is in append mode, otherwise \c false.
    //! \param[in] policy       Defined where a frame starts.
    //! \note In append mode the frame index will fixed by reading the stream.
    //----------------------------------------
    virtual bool open(IoStreamPtr&& ioStream, const bool appendMode, const FramingPolicyIn6130& policy) = 0;

public: // implements DataPackageStreamWriter
    //========================================
    //! \brief Request resource access to append data if possible and takeover the stream ownership.
    //! \note Supports only the uri protocols UriProtocol::File and UriProtocol::NoAddr.
    //! \param[in] ioStream     Resource stream handle.
    //! \param[in] append       \c true if the resource requested is in append mode, otherwise \c false.
    //----------------------------------------
    bool open(IoStreamPtr&& ioStream, const bool appendMode) override
    {
        return this->open(std::move(ioStream), appendMode, FramingPolicyIn6130::getDefaultFramingPolicy());
    }
};

//==============================================================================

//========================================
//! \brief Nullable IdcDataPackageStreamWriter pointer.
//----------------------------------------
using IdcDataPackageStreamWriterPtr = std::unique_ptr<IdcDataPackageStreamWriter>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
