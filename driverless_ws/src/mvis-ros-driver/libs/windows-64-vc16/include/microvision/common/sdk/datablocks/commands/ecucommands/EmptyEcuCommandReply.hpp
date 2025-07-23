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
//! \date Feb 14, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/commands/ecucommands/EcuCommand.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<CommandId::Id cmdId>
class EmptyEcuCommandReply final : public EcuCommandReply<cmdId>, public SpecializedDataContainer
{
public:
    //========================================
    //! \brief The container type which is used to compute the hash.
    //--------------------------------------
    static constexpr const char* containerType{"sdk.emptyecucommandreply"};

    //========================================
    //! \brief Length of the reply.
    //----------------------------------------
    static const int replySize = 2;

    //========================================
    //! \brief Return the static container hash of this class.
    //! \returns Container hash.
    //--------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    EmptyEcuCommandReply() : EcuCommandReply<cmdId>() {}
    ~EmptyEcuCommandReply() override = default;

public: // implements DataContainerBase
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Deserialize data from the given stream \a is into
    //!        this CommandSetFilter.
    //! \param[in, out] is  Stream that provides the serialized
    //!                     data to fill this CommandSetFilter.
    //!                     On exit the \a is get pointer will
    //!                     be behind the read data.
    //! \param[in]      dh  IdcDataHeader that has been received
    //!                     together with the serialized data in \a is.
    //! \return Whether the operation was successful.
    //! \retval true Everything is alright, no error occurred.
    //! \retval false Reading the data from the stream has failed.
    //----------------------------------------
    virtual bool deserialize(std::istream& is, const IdcDataHeader& dh)
    {
        const int64_t startPos = streamposToInt64(is.tellg());

        readBE(is, CommandReplyBase::m_commandId);

        return !is.fail() && ((streamposToInt64(is.tellg()) - startPos) == this->getSerializedSize())
               && this->getSerializedSize() == dh.getMessageSize();
    }

    //========================================
    //! \brief Serialize data into the given stream \a os.
    //! \param[out] os  Stream that receive the serialized
    //!                 data from this CommandSetFilter.
    //! \return Whether the operation was successful.
    //! \retval true Everything is alright, no error occurred.
    //! \retval false Writing the data into the stream has failed.
    //----------------------------------------
    virtual bool serialize(std::ostream& os) const
    {
        const int64_t startPos = streamposToInt64(os.tellp());

        writeBE(os, CommandReplyBase::m_commandId);

        return !os.fail() && ((streamposToInt64(os.tellp()) - startPos) == this->getSerializedSize());
    }

    //========================================
    //! \brief Get the DataType of this DataContainer.
    //! \return Always DataType#DataType_Command.
    //----------------------------------------
    virtual DataTypeId getDataType() const { return DataTypeId::DataType_Reply2020; }

    //========================================
    //! \brief Get the size of the serialization.
    //! \return Number of bytes used by the serialization
    //!         of this Command.
    //----------------------------------------
    virtual std::streamsize getSerializedSize() const { return std::streamsize(replySize); }

public:
    bool deserializeFromStream(std::istream& is, const IdcDataHeader& dh) override { return deserialize(is, dh); }
}; // EmptyEcuCommandReply

//==============================================================================

using ReplyEcuAppBaseCtrl = EmptyEcuCommandReply<CommandId::Id::CmdManagerAppBaseCtrl>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
