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
//! \date Feb 9, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/SpecialExporterBase.hpp>

#include <microvision/common/sdk/CommandId.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <boost/unordered_map.hpp>
#include <boost/iostreams/stream.hpp>

#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif // gcc or clang

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CommandCBase
{
public:
    //========================================
    //!\brief Used by templates to identify this
    //!       class when receiving derived classes.
    //!
    //! CommandCBase alias CommonBase is the class
    //! to be shared by all derived special command
    //! classes.
    //! The CommonBase alias is used in templates
    //! as ImporterProvider to identify
    //! the correct map.
    //----------------------------------------
    using CommonBase = CommandCBase;

    //========================================
    //!\brief The numeric type which used to
    //!       distinguish between the various
    //!       special command classes.
    //----------------------------------------
    using KeyType = CommandId::Id;

public:
    //========================================
    //! \brief Create a Command object.
    //! \param[in] commandId      Id of the command.
    //----------------------------------------
    explicit CommandCBase(const CommandId commandId) : m_commandId(CommandId(commandId)) {}
    virtual ~CommandCBase() {}

public:
    virtual uint64_t getClassIdHash() const = 0;

protected:
    CommandCBase();

public:
    //========================================
    //! \brief Get the id of this Command.
    //! \return the id of this Command.
    //----------------------------------------
    CommandId getCommandId() const { return m_commandId; }

protected:
    //========================================
    //! \brief The id of this Command.
    //----------------------------------------
    CommandId m_commandId;
}; // CommandCBase

//==============================================================================

//==============================================================================

//! \brief FUSION SYSTEM/ECU Set Filter Command
//!
//! Before the FUSION SYSTEM or ECU sends data after connecting to it (default port 12002), a filter command must be sent.
//!
//! For sending commands to the MVIS LUX the data type 0x2010 is used.
//! The MVIS LUX replies to a command with a dedicated reply message.
class Command2010 : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;
    friend class SpecialCommand2010Importer2010;

public:
    using SpecialCommandMap = boost::unordered_map<size_t, std::shared_ptr<CommandCBase>>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.command2010"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    Command2010() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    const CommandId getCommandId() const { return m_commandId; }
    const std::vector<char>& getRawContent() const { return m_rawContent; }
    const SpecialCommandMap& getSpecialCommand() const { return m_specialCommandMap; }

public:
    bool setSpecialCommand(const CommandCBase& cmd, const SpecialExporterBase<CommandCBase>& exporter)
    {
        const std::streamsize serializedSize = exporter.getSerializedSize(cmd);
        m_rawContent.resize(std::size_t(serializedSize));
        boost::iostreams::stream<boost::iostreams::array> strm(m_rawContent.data(), serializedSize);
        std::ostream& s = static_cast<std::ostream&>(strm);
        const bool ok   = exporter.serialize(s, cmd);

        //m_specialCommandMap.

        return ok;
    }

protected:
    std::vector<char> m_rawContent{};
    CommandId m_commandId{CommandId::Id(0)};
    //========================================
    //!\brief If not 0, pointer to an instance of the
    //!       special command.
    //!\attention Command2010Container does not take
    //!           the ownership of this instance.
    //!           Hence it must not delete this
    //!           pointer on destruction.
    //----------------------------------------
    SpecialCommandMap m_specialCommandMap{};
}; // Command2010

//==============================================================================
//==============================================================================
//==============================================================================

inline bool operator==(const CommandCBase& lhs, const CommandCBase& rhs)
{
    return CommandId::Id(lhs.getCommandId()) == CommandId::Id(rhs.getCommandId());
}

//==============================================================================

inline bool operator!=(const CommandCBase& lhs, const CommandCBase& rhs)
{
    return CommandId::Id(lhs.getCommandId()) != CommandId::Id(rhs.getCommandId());
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================

#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif // gcc or clang

//==============================================================================
