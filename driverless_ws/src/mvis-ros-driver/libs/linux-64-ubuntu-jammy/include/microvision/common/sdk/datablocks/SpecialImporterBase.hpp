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
//! \date Sep 12, 2017
//!
//!Importer Base class
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/misc/SdkExceptions.hpp>
#include <microvision/common/logging/logging.hpp>

#include <boost/function.hpp>

#include <cstddef> // for std::streamsize

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class DataContainerBase;
class DataContainerListenerBase;

//==============================================================================

template<typename SpecialContainerCommonBase>
class SpecialImporterBase
{
public:
    class ImporterRegisterId;

public:
    SpecialImporterBase() {}
    SpecialImporterBase(const SpecialImporterBase&) = delete;
    SpecialImporterBase& operator=(const SpecialImporterBase&) = delete;

    virtual ~SpecialImporterBase() {}

public:
    //========================================
    //! \brief Get the DataType of exporter/importer.
    //! \return The DataTypeId of the data this exporter/importer
    //!         can handle.
    //----------------------------------------
    virtual DataTypeId getDataType() const = 0;

    //========================================
    //! \brief Get serializable size of data from exporter/importer.
    //! \return Number of Bytes used by data type.
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const SpecialContainerCommonBase& s) const = 0;

    virtual bool deserialize(std::istream& stream, SpecialContainerCommonBase& s, const IdcDataHeader& d) const = 0;

public:
    virtual std::shared_ptr<SpecialContainerCommonBase> createContainer() const = 0;

public:
    virtual bool callListener(DataContainerListenerBase* l, const SpecialContainerCommonBase& s) const = 0;

protected:
    static constexpr const char* loggerId = "microvision::common::sdk::SpecialImporterBase";
    microvision::common::logging::LoggerSPtr m_logger{
        microvision::common::logging::LogManager::getInstance().createLogger(loggerId)};

}; // SpecialImporterBase

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================

template<typename SpecialContainerCommonBase>
class SpecialImporterBase<SpecialContainerCommonBase>::ImporterRegisterId
{
public:
    using Key                    = std::pair<typename SpecialContainerCommonBase::KeyType, std::size_t>;
    using ImporterCreateFunction = boost::function<SpecialImporterBase*()>;

public:
    ImporterRegisterId(const Key key, ImporterCreateFunction importerCreate)
      : m_key(key), m_importerCreate(importerCreate)
    {}

public:
    Key getKey() const { return m_key; }
    ImporterCreateFunction getValue() const { return m_importerCreate; }

private:
    Key m_key;
    ImporterCreateFunction m_importerCreate;
}; // ImporterRegisterId

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
