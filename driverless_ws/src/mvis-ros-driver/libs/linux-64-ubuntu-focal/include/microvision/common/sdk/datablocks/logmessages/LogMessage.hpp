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
//! \date May 22, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessage64x0Base.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief General data container for log messages
//!
//! Special data type:
//! \ref microvision::common::sdk::LogMessage64x0Base
//! \ref microvision::common::sdk::LogMessageDebug6430
//! \ref microvision::common::sdk::LogMessageError6400
//! \ref microvision::common::sdk::LogMessageNote6420
//! \ref microvision::common::sdk::LogMessageWarning6410
//------------------------------------------------------------------------------
class LogMessage final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;
    template<class ContainerType>
    friend class LogMessageExporter;
    template<class ContainerType>
    friend class LogMessageImporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.LogMessage"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    LogMessage() : DataContainerBase(){};

    //========================================
    //! \brief Constructor.
    //----------------------------------------
    LogMessage(const std::string& msg, const LogMessage64x0Base::TraceLevel& tlevel)
    {
        m_message       = msg;
        m_msgTraceLevel = tlevel;
    };

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~LogMessage() = default;

public:
    //========================================
    //!\brief Returns the trace level of the message.
    //!       Possible trace levels are Off, Error, Warning, Note, Debug
    //!\return TraceLevel.
    //----------------------------------------
    LogMessage64x0Base::TraceLevel getTraceLevel() const { return m_msgTraceLevel; };

    //========================================
    //!\brief Get the content of the message.
    //----------------------------------------
    const std::string& getMessage() const { return m_message; }

public:
    //========================================
    //!\brief Set the trace level of the message.
    //----------------------------------------
    void setTraceLevel(LogMessage64x0Base::TraceLevel&& newTraceLevel) { m_msgTraceLevel = newTraceLevel; }

    //========================================
    //!\brief Set the content of the message.
    //----------------------------------------
    void setMessage(const std::string& newMessage) { m_message = newMessage; }

    //========================================
    //!\brief Set the content of the message.
    //----------------------------------------
    void setMessage(std::string&& newMessage) { m_message = newMessage; }

public:
    //========================================
    //! \brief Convert all letters in \a inStr to ASCII letters.
    //! \param[in] inStr  A vector containing characters to be
    //!                   converted to a string that contains only
    //!                   ASCII characters.
    //! \return The ASCII version of the string given in \a inStr.
    //----------------------------------------
    static std::string toASCII(const std::vector<char>& inStr)
    {
        std::string outStr;
        std::transform(inStr.begin(), inStr.end(), std::back_inserter(outStr), toascii);
        return outStr;
    }

    //========================================
    //! \brief Convert a uint8_t to a TraceLevel
    //! \param[in] tl  A uint8 that want to be transformed.
    //! \return The corresponding TraceLevel
    //----------------------------------------
    LogMessage64x0Base::TraceLevel convert(uint8_t tl)
    {
        switch (tl)
        {
        case 0:
            return LogMessage64x0Base::TraceLevel::Off;
        case 1:
            return LogMessage64x0Base::TraceLevel::Error;
        case 2:
            return LogMessage64x0Base::TraceLevel::Warning;
        case 3:
            return LogMessage64x0Base::TraceLevel::Note;
        case 4:
            return LogMessage64x0Base::TraceLevel::Debug;
        default:
            return LogMessage64x0Base::TraceLevel::Off;
        }
    }

protected:
    std::string m_message; //< Content of this LogMessage.
    LogMessage64x0Base::TraceLevel m_msgTraceLevel; //< TraceLevel of this LogMessage.
}; // LogMessageContainer

//==============================================================================

bool operator==(const LogMessage& lhs, const LogMessage& rhs);
bool operator!=(const LogMessage& lhs, const LogMessage& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
