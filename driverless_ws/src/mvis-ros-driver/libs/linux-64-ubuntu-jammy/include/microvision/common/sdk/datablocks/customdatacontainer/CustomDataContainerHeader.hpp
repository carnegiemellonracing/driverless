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
//! \date Sep 2nd, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/customdatacontainer/CustomDataContainerBase.hpp>

#include <string>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This header contains idc custom datatype header serialization functionality.
//!
//! Contains a uuid/GUID, name, description and content length.
//!
//! \note Use these functions in your own custom data containers to serialize the header!
//------------------------------------------------------------------------------
class CustomDataContainerHeader
{
private:
    static constexpr uint8_t lengthOfUuid = 16; // a uuid/GUID is stored in 16 bytes (see boost::uuid)

public:
    //========================================
    //! \brief Calculate the size of the serialized custom data container header.
    //!
    //! \param[in]  name  Human readable informative name
    //! \param[in] desc   Description of the usage of this data container
    //----------------------------------------
    static std::streamsize getSerializedSize(const std::string& name, const std::string& desc)
    {
        return static_cast<std::streamsize>(lengthOfUuid //
                                            + 4 // length of name in chars
                                            + name.length() // name characters
                                            + 4 // length of description in chars
                                            + desc.length() // description characters
                                            + 4 // content length in bytes
        );
    }

    //========================================
    //! \brief Write the standard custom data container header.
    //!
    //! This header contains the unique id, a name and description
    //! for the custom data container and the size of the content payload.
    //!
    //! \param[in, out] os        Output stream receiving the header data.
    //! \param[in] uuid           Unique id
    //! \param[in] name           Human readable informative name
    //! \param[in] desc           Description of the usage of this data container
    //! \param[in] contentlength  Length of content following in this container
    //----------------------------------------
    static void serialize(std::ostream& os,
                          const DataContainerBase::Uuid& uuid,
                          const std::string& name,
                          const std::string& desc,
                          uint32_t contentlength);

    //========================================
    //! \brief Peek the header for a custom data container uuid.
    //!
    //! This header contains the unique id as the first bytes of its payload.
    //!
    //! \param[in] data           Serialized header bytes.
    //! \return Custom data type unique id.
    //!
    //! \note This does not verify the header information.
    //----------------------------------------
    static inline DataContainerBase::Uuid peekUuid(const char* data)
    {
        return *reinterpret_cast<const DataContainerBase::Uuid*>(data);
    }

    //========================================
    //! \brief Read in the header for a custom data container.
    //!
    //! This header contains the unique id, a name and description for the custom data container
    //! and the size of the content payload.
    //!
    //! \param[in] is             Input stream containing the header data.
    //! \param[in] uuid           Unique id
    //! \param[in] name           Human readable informative name
    //! \param[in] desc           Description of the usage of this data container
    //! \return Length of content following in this container
    //!
    //! \note This does not verify the header information.
    //! \note Most custom data containers will not need this information because it is static for them.
    //----------------------------------------
    static uint32_t deserialize(std::istream& is, std::string& uuid, std::string& name, std::string& desc);

    //========================================
    //! \brief Read in the header for a custom data container.
    //!
    //! This header contains the unique id, a name and description for the custom data container
    //! and the size of the content payload.
    //!
    //! \param[in] is  Input stream containing the header data.
    //! \return Size of content to follow in the stream.
    //!
    //! \note This does not verify the header information.
    //----------------------------------------
    static uint32_t deserialize(std::istream& is);

    //========================================
    //! \brief Read in the header for a custom data container if matching given uuid.
    //!
    //! This header contains the unique id, a name and description for the custom data container
    //! and the size of the content payload. Header is completely read from the stream if matching.
    //!
    //! \tparam[in] T    Wanted custom data container. Read fails if idc data type is not matching.
    //! \param[in]  is   Input stream containing the header data.
    //! \return \c True if custom data container header matches given uui, \c false if not.
    //!
    //! \note This does not verify the header information except for comparing the expected uuid.
    //!       If failed check only uuid is read from stream (not full header).
    //----------------------------------------
    template<typename T>
    static bool deserializeCheck(std::istream& is)
    {
        // uuid
        DataContainerBase::Uuid inUuid;
        is.read(reinterpret_cast<char*>(inUuid.data), lengthOfUuid);
        if (inUuid != T::getUuidStatic())
        {
            return false;
        }

        // skip name
        uint32_t size;
        readBE(is, size);
        is.seekg(size, std::ios::cur);

        // skip description
        readBE(is, size);
        is.seekg(size, std::ios::cur);

        // skip size
        readBE(is, size);

        return true;
    }
}; // CustomDataContainerHeader

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
