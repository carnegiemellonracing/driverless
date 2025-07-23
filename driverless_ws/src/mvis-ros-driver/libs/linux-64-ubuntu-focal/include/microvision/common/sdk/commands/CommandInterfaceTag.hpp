//==============================================================================
//! \file
//!
//! \brief Interface for all devices which support commands.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 7th, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandInterface;

//==============================================================================
//! \brief Can only be constructed by an CommandInterface object.
//! An CommandInterfaceTag instance monitors whether the originating
//! CommandInterface instance has been destructed
//------------------------------------------------------------------------------
class CommandInterfaceTag
{
    //========================================
    //! \brief CommandInterface is the only class which can construct CommandInterfaceTag instances
    //--------------------------------------
    friend class CommandInterface;

private:
    //========================================
    //! \brief Constructor
    //! \param[in]  isAlive  The internal boolean shared pointer of the monitored CommandInterface instance.
    //! \param[out] originator Pointer to the monitored command interface.
    //--------------------------------------
    CommandInterfaceTag(const std::shared_ptr<bool> isAlive, CommandInterface* originator);

public:
    //========================================
    //! \brief !operator.
    //--------------------------------------
    bool operator!() const;

    //========================================
    //! \brief Boolean conversion operator.
    //--------------------------------------
    operator bool() const;

    //========================================
    //! \brief *operator
    //--------------------------------------
    CommandInterface& operator*();

    //========================================
    //! \brief ->operator
    //--------------------------------------
    CommandInterface* operator->();

public:
    //========================================
    //! \brief Checks whether the monitored CommandInterface instance has been destructed.
    //! \returns \c True, if the monitored CommandInterface instance has been destructed, false otherwise.
    //--------------------------------------
    bool expired() const;

private:
    //========================================
    //! \brief Logger
    //--------------------------------------
    static MICROVISION_SDK_API logging::LoggerSPtr logger;

private:
    //========================================
    //! \brief Indicates the expiration status of the monitored CommandInterface instance.
    //! If this weak_ptr is expired, it means that the monitored instance has been destructed.
    //--------------------------------------
    std::weak_ptr<bool> m_originatorIsAlive;

    //========================================
    //! \brief The monitored CommandInterface instance.
    //--------------------------------------
    CommandInterface* m_originator;
};

using CommandInterfaceTagUPtr = std::unique_ptr<CommandInterfaceTag>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
