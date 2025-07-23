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
//! \date Apr 8, 2015
//------------------------------------------------------------------------------
#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/commands/ecucommands/EcuCommand.hpp>
#include <microvision/common/sdk/datablocks/commands/ecucommands/EcuCommandC.hpp>
#include <microvision/common/sdk/datablocks/commands/ecucommands/AppBaseStatusDefinitions.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/DataTypeId.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================
//!\class ReplyManagerAppBaseStatus
//!\brief Ecu AppBase status command
//!\date Apr 7, 2015
//------------------------------------------------------------------------------
class ReplyEcuAppBaseStatus : public EcuCommandReply<CommandId::Id::CmdManagerAppBaseStatus>,
                              public AppBaseStatusDefinitions,
                              public SpecializedDataContainer
{
public:
    enum class AppBaseStatusId : uint8_t
    {
        Recording = 0x01
    };

public:
    //========================================
    //! \brief The container type which is used to compute the hash.
    //--------------------------------------
    static constexpr const char* containerType{"sdk.replyecuappbasestatus"};

    //========================================
    //! \brief Length of the CommandManagerAppBaseStatus command.
    //----------------------------------------
    static const int commandSizeBase = 4;

    //========================================
    //! \brief Return the static container hash of this class.
    //! \returns Container hash.
    //--------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

    //========================================
    //! \brief Maximum size of the data string.
    //--------------------------------------
    static constexpr uint16_t maxDataStringSize{10000};

public:
    ReplyEcuAppBaseStatus();

public: // implements DataContainerBase
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); };

public:
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
    virtual std::streamsize getSerializedSize() const { return std::streamsize(commandSizeBase + m_data.size()); }

    //========================================
    //! \brief Deserialize data from the given stream \a is into
    //!        this CommandManagerAppBaseStatusReply.
    //! \param[in, out] is  Stream that provides the serialized
    //!                     data to fill this CommandManagerAppBaseStatusReply.
    //!                     On exit the \a is get pointer will
    //!                     be behind the read data.
    //! \param[in]      dh  IdcDataHeader that has been received
    //!                     together with the serialized data in \a is.
    //! \return Whether the operation was successful.
    //! \retval true Everything is alright, no error occurred.
    //! \retval false Reading the data from the stream has failed.
    //----------------------------------------
    virtual bool deserialize(std::istream& is, const IdcDataHeader& dh);

    //========================================
    //! \brief Serialize data into the given stream \a os.
    //! \param[out] os  Stream that receive the serialized
    //!                 data from this CommandManagerAppBaseStatusReply.
    //! \return Whether the operation was successful.
    //! \retval true Everything is alright, no error occurred.
    //! \retval false Writing the data into the stream has failed.
    //----------------------------------------
    virtual bool serialize(std::ostream& os) const;

public:
    bool deserializeFromStream(std::istream& is, const IdcDataHeader& dh) override { return deserialize(is, dh); }

public:
    AppBaseStatusId getStatusId() { return m_statusId; }

    //========================================
    //! \brief The status of the ECU recording.
    //! \return The status of the ECU recording.
    //! \retval '' The recording is not available,
    //!            an empty string will be returned.
    //! \retval '0' The recording is available but not active.
    //! \retval '1' The recording is available and active.
    //----------------------------------------
    std::string getData() { return m_data; }

protected:
    AppBaseStatusId m_statusId;
    std::string m_data;
}; // ReplyManagerAppBaseStatus

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
