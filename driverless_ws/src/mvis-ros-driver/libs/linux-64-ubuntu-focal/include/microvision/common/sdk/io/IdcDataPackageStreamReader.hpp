//==============================================================================
//! \file
//!
//! \brief Provides functionality to read a IdcDataPackage stream.
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

#include <microvision/common/sdk/listener/IdcEmitter.hpp>

#include <microvision/common/sdk/io/IdcDataPackage.hpp>
#include <microvision/common/sdk/io/DataPackageStreamReader.hpp>
#include <microvision/common/sdk/io/IdcStreamBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Provides functionality to read a IdcDataPackage stream.
//!
//! Provides functions to read forward and backward in IdcDataPackage stream.
//!
//! \extends IdcStreamBase
//! \extends DataPackageStreamReader
//! \extends IdcEmitter
//------------------------------------------------------------------------------
class IdcDataPackageStreamReader : public IdcStreamBase, public virtual DataPackageStreamReader, public IdcEmitter
{
public:
    //========================================
    //! \brief Made all open implementations visible.
    //----------------------------------------
    using DataPackageStreamReader::open;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdcDataPackageStreamReader() override = default;

public:
    //========================================
    //! \brief Peek first IdcDataHeader.
    //! \return First IdcDataHeader if found, otherwise nullptr.
    //----------------------------------------
    virtual IdcDataHeaderPtr peekFirstIdcDataHeader() = 0;

    //========================================
    //! \brief Peek last IdcDataHeader.
    //! \return Last IdcDataHeader if found, otherwise nullptr.
    //----------------------------------------
    virtual IdcDataHeaderPtr peekLastIdcDataHeader() = 0;

    //========================================
    //! \brief Peek next IdcDataHeader.
    //! \return Next IdcDataHeader if found, otherwise nullptr.
    //----------------------------------------
    virtual IdcDataHeaderPtr peekNextIdcDataHeader() = 0;

    //========================================
    //! \brief Peek previous IdcDataHeader.
    //! \return Previous IdcDataHeader if found, otherwise nullptr.
    //----------------------------------------
    virtual IdcDataHeaderPtr peekPreviousIdcDataHeader() = 0;

public:
    //========================================
    //! \brief Read first IdcDataPackage.
    //! \note Will ignored IdcTrailer and FrameIndex.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    virtual IdcDataPackagePtr readFirstIdcDataPackage() = 0;

    //========================================
    //! \brief Read last IdcDataPackage.
    //! \note Will ignored IdcTrailer and FrameIndex.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    virtual IdcDataPackagePtr readLastIdcDataPackage() = 0;

    //========================================
    //! \brief Read next IdcDataPackage.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    virtual IdcDataPackagePtr readNextIdcDataPackage() = 0;

    //========================================
    //! \brief Read previous IdcDataPackage.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    virtual IdcDataPackagePtr readPreviousIdcDataPackage() = 0;

public:
    //========================================
    //! \brief Added all deserializations of the next data package into data container by listener(s).
    //! \note The parameter dataContainers will only additive changed.
    //! \param[out] dataContainers  All convertions of the next data package.
    //! \return \c True if deserializations added to the vector, otherwise \c false.
    //----------------------------------------
    virtual bool tryGetNextDataContainers(IdcEmitter::DataContainerImporterContextList& dataContainers);

    //========================================
    //! \brief Added all deserializations of the data package into data container by listener(s).
    //! \note The parameter dataContainers will only additive changed.
    //! \param[in] dataPackage      Readed data package.
    //! \param[out] dataContainers  All convertions of the \a dataPackage.
    //! \return \c True if deserializations added to the vector, otherwise \c false.
    //----------------------------------------
    virtual bool tryGetDataContainers(const IdcDataPackagePtr& dataPackage,
                                      IdcEmitter::DataContainerImporterContextList& dataContainers);

public: // implements DataPackageStreamReader
    //========================================
    //! \brief Utility method that goes through the stream and notifies all registered
    //!        listeners without time synchronisation
    //! \return The number of data packages which been emitted to any registered listeners.
    //----------------------------------------
    uint32_t loopAndNotify() override;

    //========================================
    //! \brief Read first IdcDataPackage.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    DataPackagePtr readFirstPackage() override;

    //========================================
    //! \brief Read last IdcDataPackage.
    //! \note Will ignored IdcTrailer and FrameIndex.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    DataPackagePtr readLastPackage() override;

    //========================================
    //! \brief Read next IdcDataPackage.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    DataPackagePtr readNextPackage() override;

    //========================================
    //! \brief Read previous IdcDataPackage.
    //! \return IdcDataPackage if found, otherwise nullptr.
    //----------------------------------------
    DataPackagePtr readPreviousPackage() override;
};

//==============================================================================

//========================================
//! \brief Nullable IdcDataPackageStreamReader pointer.
//----------------------------------------
using IdcDataPackageStreamReaderPtr = std::unique_ptr<IdcDataPackageStreamReader>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
