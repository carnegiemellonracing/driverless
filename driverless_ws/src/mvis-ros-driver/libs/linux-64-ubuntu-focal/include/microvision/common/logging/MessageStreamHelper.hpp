//==============================================================================
//! \file
//! \brief Helper for using streams with the Message class.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) Microvision 2010-2024
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! MicroVisionLicense.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Oct 05, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/logging/Message.hpp>

//==============================================================================

// Define calling convention for IO manipulators.
#ifdef _WIN32
#    define LOGGING_IOMANIP_CALL_DECL __cdecl
#else
#    define LOGGING_IOMANIP_CALL_DECL
#endif // _WIN32

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief Helper class for using streaming output operator ('<<') with log messages.
//------------------------------------------------------------------------------
template<typename T, typename StorageT = const T>
class MessageStreamHelper : public Message
{
public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    MessageStreamHelper(const T value) : Message(), m_value(value) {}

public:
    //========================================
    //! \brief Print the value to the given stream.
    //!
    //! \param[in,out] ostr  Stream to print the value to.
    //----------------------------------------
    void print(std::ostream& ostr) const override { ostr << m_value; }

private:
    StorageT m_value;
}; // MessageStreamHelper

//==============================================================================
//! \brief Implementation of streaming output operator ('<<') for a log message using simple types.
//!
//! \tparam T           Type of value to be streamed.
//! \param[in,out] msg  Log message to stream to.
//! \param[in] v        Value to be streamed.
//! \return Log message as given in parameter \a msg.
//------------------------------------------------------------------------------
template<typename T>
inline MessageSPtr operator<<(const MessageSPtr msg, const T v)
{
    static_assert(std::is_pointer<T>() == false, "Pointers need special treatment!");

    MessageSPtr helper = std::make_shared<MessageStreamHelper<T>>(v);
    msg->addHelper(helper);
    return msg;
}

//==============================================================================
//! \brief Implementation of streaming output operator ('<<') for a log message using type const char*.
//!
//! \param[in,out] msg  Log message to stream to.
//! \param[in] v        Value to be streamed.
//! \return Log message as given in parameter \a msg.
//------------------------------------------------------------------------------
template<>
inline MessageSPtr operator<<(const MessageSPtr msg, const char* const v)
{
    // Use std::string as storage to avoid the pointer going out of scope when logging asynchronously.
    MessageSPtr helper = std::make_shared<MessageStreamHelper<std::string>>(v);
    msg->addHelper(helper);
    return msg;
}

//==============================================================================
//! \brief Implementation of streaming output operator ('<<') for a log message using type char*.
//!
//! \param[in,out] msg  Log message to stream to.
//! \param[in] v        Value to be streamed.
//! \return Log message as given in parameter \a msg.
//------------------------------------------------------------------------------
template<>
inline MessageSPtr operator<<(const MessageSPtr msg, char* const v)
{
    return msg << static_cast<const char*>(v);
}

//==============================================================================
//! \brief Implementation of streaming output operator ('<<') for a log message using type const void*.
//!
//! \param[in,out] msg  Log message to stream to.
//! \param[in] v        Value to be streamed.
//! \return Log message as given in parameter \a msg.
//------------------------------------------------------------------------------
template<>
inline MessageSPtr operator<<(const MessageSPtr msg, const void* const v)
{
    // Use std::string as storage to avoid the pointer going out of scope when logging asynchronously.
    MessageSPtr helper = std::make_shared<MessageStreamHelper<intptr_t>>(reinterpret_cast<intptr_t>(v));
    msg->addHelper(helper);
    return msg;
}

//==============================================================================
//! \brief Implementation of streaming output operator ('<<') for a log message using type char*.
//!
//! \param[in,out] msg  Log message to stream to.
//! \param[in] v        Value to be streamed.
//! \return Log message as given in parameter \a msg.
//------------------------------------------------------------------------------
template<>
inline MessageSPtr operator<<(const MessageSPtr msg, void* const v)
{
    return msg << static_cast<const void*>(v);
}

//==============================================================================
//! \brief Implementation of streaming output operator ('<<') for a log message using IO manipulators.
//!
//! \tparam TReturn     Return value type of IO manipulator function.
//! \tparam TStream     Stream type of IO manipulator function.
//! \param[in,out] msg  Log message to stream to.
//! \param[in] man      IO manipulator function.
//! \return Log message as given in parameter \a msg.
//------------------------------------------------------------------------------
template<typename TReturn, typename TStream>
inline MessageSPtr operator<<(const MessageSPtr msg, TReturn&(LOGGING_IOMANIP_CALL_DECL* man)(TStream&))
{
    typedef TReturn&(LOGGING_IOMANIP_CALL_DECL * ManipType)(TStream&);

    MessageSPtr helper = std::make_shared<MessageStreamHelper<ManipType, ManipType>>(man);
    msg->addHelper(helper);

    // mark invalid for async printing because the manipulator might be invalid at that time already!
    msg->enableTextPreparation();

    return msg;
}

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
