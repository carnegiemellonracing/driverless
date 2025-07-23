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
//! \date Feb 13, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/SpecialImporterBase.hpp>

#include <boost/unordered_map.hpp>

#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<typename Identifier>
class ImporterProviderGlobal final
{
public:
    using KeyType                = typename Identifier::KeyType;
    using ImporterRegisterId     = typename SpecialImporterBase<typename Identifier::CommonBase>::ImporterRegisterId;
    using ImporterCreateFunction = typename ImporterRegisterId::ImporterCreateFunction;

    using GlobalProviderMap = boost::unordered_map<typename ImporterRegisterId::Key, ImporterCreateFunction>;

private:
    ImporterProviderGlobal() : m_glProviderMap() {}
    ~ImporterProviderGlobal() = default;

private:
    ImporterProviderGlobal(const ImporterProviderGlobal& other) = delete;
    ImporterProviderGlobal& operator=(const ImporterProviderGlobal& other) = delete;

public:
    static ImporterProviderGlobal<Identifier>& getInstance()
    {
        static ImporterProviderGlobal<Identifier> dbbp;
        return dbbp;
    }

public:
    const ImporterRegisterId& registerImporter(const ImporterRegisterId& dbri)
    {
        m_glProviderMap[dbri.getKey()] = dbri.getValue();
        return dbri;
    }

    const GlobalProviderMap& getMap() const { return m_glProviderMap; }

protected:
    GlobalProviderMap m_glProviderMap;
}; // ImporterProviderGlobal

//==============================================================================

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================

template<typename Identifier>
class ImporterProvider
{
public:
    using KeyType = typename Identifier::KeyType;

public:
    class ImporterCreator;

    using ImporterRegisterId = typename SpecialImporterBase<Identifier>::ImporterRegisterId;
    using ProviderMap        = boost::unordered_map<typename ImporterRegisterId::Key, ImporterCreator>;

public:
    ImporterProvider(ImporterProviderGlobal<Identifier>& globalProvider)
    {
        using GlobalMap                               = typename ImporterProviderGlobal<Identifier>::GlobalProviderMap;
        const GlobalMap& globalMap                    = globalProvider.getMap();
        typename GlobalMap::const_iterator globalIter = globalMap.begin();
        for (; globalIter != globalMap.end(); ++globalIter)
        {
            m_providerMap[globalIter->first] = ImporterCreator(globalIter->second);
        } // for globalIter
    }

    ~ImporterProvider() = default;

private:
    ImporterProvider(const ImporterProvider&) = delete;
    ImporterProvider& operator=(const ImporterProvider&) = delete;

public:
    //========================================
    SpecialImporterBase<Identifier>* getImporter(const typename ImporterRegisterId::Key key)
    {
        //        LOGWARNING(logger, "Size of map: " << providerMap.size());

        typename ProviderMap::iterator iter = m_providerMap.find(key);
        if (iter == m_providerMap.end())
        {
            return nullptr;
        }

        return iter->second.getImporter();
    }

    const ProviderMap& getMap() const { return m_providerMap; }

protected:
    ProviderMap m_providerMap;
}; // ImporterProvider

//==============================================================================

template<typename Identifier>
class ImporterProvider<Identifier>::ImporterCreator
{
public:
    using ImporterCreateFunction = typename SpecialImporterBase<Identifier>::ImporterRegisterId::ImporterCreateFunction;

public:
    ImporterCreator() = default;

    explicit ImporterCreator(ImporterCreateFunction createImporter) : m_createImporter(createImporter) {}

    ~ImporterCreator() = default;

public:
    //========================================
    SpecialImporterBase<Identifier>* getImporter()
    {
        if (!m_importer && m_createImporter)
        {
            m_importer.reset(m_createImporter());
        }
        return m_importer.get();
    }

    //========================================
    ImporterCreateFunction getCreateImporterPtr() const { return m_createImporter; }

protected:
    std::shared_ptr<SpecialImporterBase<Identifier>> m_importer{};
    typename SpecialImporterBase<Identifier>::ImporterRegisterId::ImporterCreateFunction m_createImporter;
}; // ContainerImporter

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
