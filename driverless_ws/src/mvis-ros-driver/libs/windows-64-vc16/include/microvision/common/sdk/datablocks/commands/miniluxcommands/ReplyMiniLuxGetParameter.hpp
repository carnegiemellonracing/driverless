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
//! \date Apr 10, 2015
//------------------------------------------------------------------------------
#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/commands/miniluxcommands/MiniLuxCommand.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/ParameterIndex.hpp>
#include <microvision/common/sdk/misc/ParameterData.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \class ReplyMiniLuxGetParameter
//! \brief
//! \date Apr 10, 2015
//------------------------------------------------------------------------------
class ReplyMiniLuxGetParameter
  : public microvision::common::sdk::MiniLuxCommandReply<CommandId::Id::CmdLuxGetParameter>,
    public microvision::common::sdk::SpecializedDataContainer
{
public:
    //========================================
    //! \brief Length of the CommandManagerAppBaseStatus command.
    //----------------------------------------
    static const int replySize = 8;

public:
    ReplyMiniLuxGetParameter();

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

public:
    ParameterIndex getParameterIndex() const { return m_parameterIndex; }
    ParameterData getParameterData() const { return m_parameterData; }

protected:
    ParameterIndex m_parameterIndex;
    ParameterData m_parameterData;
}; // ReplyMiniLuxGetParameter

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
