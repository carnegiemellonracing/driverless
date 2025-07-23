//==============================================================================
//! \file
//! \brief Message to be logged.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) Microvision 2010-2024
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! MicroVisionLicense.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/logging/LoggingExport.hpp>

#include <chrono>
#include <string>
#include <ostream>
#include <sstream>
#include <memory>
#include <vector>
#include <cstdint>
#include <list>
#include <thread>

//==============================================================================

#ifdef _WIN32
#    pragma warning(disable : 4251)
// class 'xxx' needs to have dll - interface to be used by clients of class 'yyy'
#endif

//==============================================================================
namespace microvision {
namespace common {
namespace logging {
//==============================================================================

//==============================================================================
//! \brief This class is used for transporting log messages between front- and backends.
//!
//! Simple log messages containing a string only can be directly created using the constructor. For more complex
//! messages, it is possible to concatenate fields using the streaming output operator ('<<'). Consider using the
//! \a LOGMSG or \a LOGMSG_TEXT macro for automatically populating the source code information in the message.
//!
//! \note This class is not thread-safe. An application has to take special actions if instances are shared between
//!       different threads.
//------------------------------------------------------------------------------
class LOGGING_EXPORT Message
{
    template<typename T>
    friend std::shared_ptr<Message> operator<<(const std::shared_ptr<Message> msg, const T v);

    template<typename R, typename S>
    friend std::shared_ptr<Message> operator<<(const std::shared_ptr<Message> msg, R& (*man)(S&));

public:
    using Timepoint = std::chrono::system_clock::time_point;

public:
    //========================================
    //! \brief Constructor.
    //!
    //! \param[in] lineNo    Line number inside the source code file where the log message is created.
    //! \param[in] function  Name of the function where the log message is created.
    //! \param[in] file      Name of the source code file where the log message is created.
    //! \param[in] text      String used a log message.
    //----------------------------------------
    Message(int lineNo, const char* function, const char* file, const std::string& text = "");

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~Message() = default;

public:
    //========================================
    //! \brief Get the timstamp when the message was created.
    //!
    //! \return Timestamp when the message was created.
    //----------------------------------------
    virtual const Timepoint& getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief Get the name of the source code file where the message was created.
    //!
    //! \return The name of the source code file where the message was created.
    //----------------------------------------
    virtual const std::string& getFile() const { return m_file; }

    //========================================
    //! \brief Get the name of the function where the message was created.
    //!
    //! \return The name of the function where the message was created.
    //----------------------------------------
    virtual const std::string& getFunction() const { return m_function; }

    //========================================
    //! \brief Get the line number in the source code file where the message was created.
    //!
    //! \return The line number in the source code file where the message was created.
    //----------------------------------------
    virtual int getLineNo() const { return m_lineNo; }

    //========================================
    //! \brief Get the log message text.
    //!
    //! \return The log message text is made of the text parameter in the constructor and optionally other parameters
    //! appended to this message using the streaming output operator ('<<').
    //----------------------------------------
    virtual std::string getText() const;

    //========================================
    //! \brief Get the sequence number.
    //!
    //! \return The sequence number.
    //!
    //! \note The sequence number is an indicator for the order of the message creation across multiple threads. It is
    //! unique as long as there is no overflow in uint64_t (~10^19).
    //----------------------------------------
    virtual uint64_t getSequenceNumber() const { return m_sequenceNumber; }

    //========================================
    //! \brief Get ID of the thread which created the message.
    //!
    //! \return The ID of the thread which created the message.
    //----------------------------------------
    virtual const std::thread::id& getThreadId() const { return m_threadId; }

    //========================================
    //! \brief Prepare this message to be scheduled asynchronously.
    //!
    //! In some cases message text printing needs to be done before scheduling (logging is still done async).
    //----------------------------------------
    void prepareText();

    //========================================
    //! \brief Print the log message text to the given stream.
    //!
    //! \param[in,out] ostr  Stream to print the message to.
    //!
    //! This method prints the text parameter given in the constructor and optionally other parameters appended to
    //! this message using the streaming output operator ('<<').
    //----------------------------------------
    virtual void print(std::ostream& ostr) const;

protected:
    //========================================
    //! \brief Get the next available sequence number.
    //!
    //! \return The next available sequence number.
    //----------------------------------------
    static uint64_t getNextSequenceNumber();

    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Message() = default;

    //========================================
    //! \brief Add a helper class to the internal list.
    //!
    //! \param[in] helper  Helper class for using streaming output operator ('<<') with log messages.
    //----------------------------------------
    void addHelper(std::shared_ptr<Message> helper);

    //========================================
    //! \brief Mark this message to be printed before scheduling for asynchronous printing.
    //!
    //! This is required for example if io manipulators are used which could be invalid/released
    //! already when the async processing thread wishes to do the printing later (plugin unload).
    //----------------------------------------
    void enableTextPreparation() { m_doAsyncStringConcatenation = false; }

private:
    int m_lineNo{0};
    std::string m_function{""};
    std::string m_file{""};
    std::string m_text{""};
    Timepoint m_timestamp{};
    uint64_t m_sequenceNumber{0};
    std::thread::id m_threadId;
    std::vector<std::shared_ptr<Message>> m_streamHelpers;
    bool m_doAsyncStringConcatenation{true}; // if true this log text string can be built in a background thread
    std::string m_preparedText{""}; // storage for the prepared log text string to be printed later
}; // Message

//==============================================================================

using MessageSPtr = std::shared_ptr<Message>;

//==============================================================================
//! \brief This functions adds a new line to the log message.
//------------------------------------------------------------------------------
inline std::ostream& endl(std::ostream& __os) { return flush(__os.put(__os.widen('\n'))); }

//==============================================================================
} // namespace logging
} // namespace common
} // namespace microvision
//==============================================================================
