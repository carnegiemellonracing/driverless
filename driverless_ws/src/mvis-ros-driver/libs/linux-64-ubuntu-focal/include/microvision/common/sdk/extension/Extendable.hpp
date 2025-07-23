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

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <set>
#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Helper class to implement an extension point for plugins and modules.
//!
//! \tparam TExtension  Type of extensions that can be registered with this extendable.
//------------------------------------------------------------------------------
template<typename TExtension>
class Extendable
{
public:
    //========================================
    //! \brief Extension pointer
    //----------------------------------------
    using TExtensionPtr = std::shared_ptr<TExtension>;

public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~Extendable() = default;

public:
    //========================================
    //! \brief Get's all extension pointers.
    //!
    //! \return Set of extension pointers.
    //----------------------------------------
    virtual const std::set<TExtensionPtr>& getExtensions(void) const { return this->m_registeredExtensions; }

    //========================================
    //! \brief Add an extension to this extension point.
    //!
    //! \param[in] ext Extension pointer.
    //----------------------------------------
    virtual const TExtensionPtr registerExtension(const TExtensionPtr& ext)
    {
        m_registeredExtensions.insert(ext);
        return ext;
    }

    //========================================
    //! \brief Remove a extension from this extension point.
    //!
    //! \param[in] ext Extension pointer.
    //----------------------------------------
    virtual void unregisterExtension(const TExtensionPtr& ext) { m_registeredExtensions.erase(ext); }

private:
    std::set<TExtensionPtr> m_registeredExtensions;
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
