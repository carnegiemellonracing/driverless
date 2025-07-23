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
//! \date Apr 5, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/logging/logging.hpp>

#include <map>
#include <exception>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Shared logger for meta information factories of all type.
//==============================================================================
class MetaInformationLogger
{
public:
    static microvision::common::logging::LoggerSPtr getLogger();

private:
    static constexpr const char* loggerId = "microvision::common::sdk::MetaInformationFactory";
};

//==============================================================================
//! \brief Factory to create meta information list entries.
//==============================================================================
template<typename TMetaInformation>
class MetaInformationFactory final : private MetaInformationLogger
{
public:
    using MetaInformationSPtr = std::shared_ptr<TMetaInformation>;
    using FactoryFunc         = MetaInformationSPtr (*)(void);

public:
    //========================================
    //! \brief Default Constructor
    //----------------------------------------
    MetaInformationFactory() = default;

public:
    //========================================
    //! \brief Register a meta information type to be available from this factory.
    //!
    //! \param[in] type  Type of meta information entry to be created.
    //! \param[in] func  Function to create.
    //----------------------------------------
    void registerType(const typename TMetaInformation::MetaInformationType& type, FactoryFunc func)
    {
        // Verify we don't overwrite anything with this name!
        if (m_factoryMap.count(type) > 0)
        {
            LOGERROR(getLogger(), "Tried to register already known type: " << type);
            return;
        }

        m_factoryMap.insert(typename FactoryMap::value_type(type, func));
    }

    //========================================
    //! \brief Register a meta information type to be available from this factory.
    //!
    //! \tparam T  Meta information bit type to be created from this registered type.
    //! \param[in] type  Type of meta information entry to be created.
    //----------------------------------------
    template<class T>
    void registerType(const typename TMetaInformation::MetaInformationType& type)
    {
        registerType(type, MetaInformationFactory::factory_function<T>);
    }

    //========================================
    //! \brief Create a meta information list entry.
    //!
    //! \param[in] type  Type of meta information entry to be created.
    //! \return Shared pointer to the created object.
    //----------------------------------------
    const MetaInformationSPtr create(const typename TMetaInformation::MetaInformationType& type) const
    {
        typename FactoryMap::const_iterator iter = m_factoryMap.find(type);
        if (iter != m_factoryMap.end())
        {
            return iter->second();
        }
        else
        {
            LOGDEBUG(getLogger(), "Wanted meta information type " << type << " unknown!");

            return nullptr;
        }
    }

private:
    using FactoryMap = std::map<typename TMetaInformation::MetaInformationType, FactoryFunc>;

private:
    //========================================
    //! \brief Template for factory function.
    //!
    //! \tparam T  Type of meta information entry to be created.
    //! \return Shared pointer to the created object.
    //----------------------------------------
    template<class T>
    static MetaInformationSPtr factory_function()
    {
        return std::make_shared<T>();
    }

private:
    FactoryMap m_factoryMap;
}; // MetaInformationFactory

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
