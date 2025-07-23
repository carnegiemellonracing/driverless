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
//! \date May 05, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/commands/ecucommands/EcuCommand.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ReplyEcuSetFilter final : public microvision::common::sdk::EcuCommandReply<CommandId::Id::CmdManagerSetFilter>,
                                public microvision::common::sdk::SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Length of the CommandManagerAppBaseStatus command.
    //----------------------------------------
    static const int replySize = 2;

    constexpr static const char* const containerType{"sdk.specialcontainer.replyecusetfilter"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ReplyEcuSetFilter();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

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
    virtual std::streamsize getSerializedSize() const { return std::streamsize(replySize); }

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
}; //ReplyEcuSetFilter

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
