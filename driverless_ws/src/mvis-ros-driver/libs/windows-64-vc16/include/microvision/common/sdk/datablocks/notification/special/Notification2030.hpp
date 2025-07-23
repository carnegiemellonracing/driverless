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
//! \date Jun 19, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Notification
//!
//! A notification is an error code that can be emitted from a Device or
//! an Interface to inform its registered message handlers about problems.
//------------------------------------------------------------------------------
class Notification2030 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    enum class TraceLevel : uint8_t
    {
        Off           = 0, //!< No trace output
        Error         = 1, //!< Show errors only (minimal trace)
        Warning       = 2, //!< Show errors and warnings
        Note          = 3, //!< Show errors, warnings and notes
        Debug         = 4, //!< Show errors, warnings, notes and debug messages
        DebugInternal = 5, //!< Show errors, warnings, notes, debug and internal framework messages (maximal trace)
        NoteConfig    = 6, //!< Show configs as lognote
        NoteTrigger   = 7, //!< Trigger message
    };

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.notification2030"};

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
    Notification2030() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~Notification2030() override = default;

public:
    //========================================
    //! \brief Get the mnemonic of the notification.
    //----------------------------------------
    const std::string& getMnemonic() const { return m_mnemonic; }

    //========================================
    //! \brief Get the severity of the notification.
    //----------------------------------------
    TraceLevel getSeverity() const { return m_severity; }

public:
    //========================================
    //! \brief Set the mnemonic of the notification.
    //----------------------------------------
    void setMnemonic(const std::string& newMnemonic) { m_mnemonic = newMnemonic; }

    //========================================
    //! \brief Set the mnemonic of the notification.
    //----------------------------------------
    void setMnemonic(std::string&& newMnemonic) { m_mnemonic = newMnemonic; }

    //========================================
    //! \brief Set the severity of the notification.
    //----------------------------------------
    void setSeverity(TraceLevel newTraceLevel) { m_severity = newTraceLevel; }

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

private:
    std::string m_mnemonic; //!< Shortname of the Content of this  notification
    TraceLevel m_severity{TraceLevel::Off}; //!< The severity level of this notification

}; // Notification2030Container

//==============================================================================

bool operator==(const Notification2030& lhs, const Notification2030& rhs);
bool operator!=(const Notification2030& lhs, const Notification2030& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
