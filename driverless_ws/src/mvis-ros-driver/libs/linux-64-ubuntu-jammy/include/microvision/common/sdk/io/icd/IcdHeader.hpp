//==============================================================================
//! \file
//!
//! \brief Extendable Iutp header.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 6th, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include "microvision/common/sdk/misc/defines/defines.hpp"

#include <microvision/common/sdk/misc/SharedBufferStream.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <vector>
#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//========================================
//! \brief Enum defining the icd header content types.
//----------------------------------------
enum class IcdHeaderType : uint16_t
{
    Invalid             = 0,
    DatatypeName        = 1,
    DatatypeVersion     = 2,
    TopicName           = 3,
    DatatypeByteOrder   = 4,
    TimeStamp           = 5,
    PayloadEASerialized = 0xFF00,
    PayloadEASerializedBE
};

//========================================
//! \brief Iutp header datatype version.
//----------------------------------------
struct IcdVersion
{
    uint16_t major;
    uint16_t minor;
};

//========================================
//! \brief Header used by Hermes bridge for transporting additional information about the icd content.
//----------------------------------------
class IcdHeader
{
public:
    using EntryList
        = std::map<IcdHeaderType,
                   SharedBuffer>; //!< List of additional entries in the header except name, version and payload.

private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::IcdHeader"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    IcdHeader();

    //========================================
    //! \brief Constructor by icd datatype name and datatype content payload size.
    //----------------------------------------
    IcdHeader(const std::string& name, const IcdVersion& version, const std::size_t size);

    //========================================
    //! \brief Compare with another header.
    //! \return Either \c true if all the same or \false if not.
    //----------------------------------------
    bool operator==(const IcdHeader& other) const;

    //========================================
    //! \brief Add an additional type of information to the header.
    //! \param[in] type   Header entry type.
    //! \param[in] value  Value to set.
    //----------------------------------------
    void setEntry(const uint16_t type, const SharedBuffer& value);

public:
    //========================================
    //! \brief Get the name of the datatype contained in the payload after this header.
    //! \return String name of data type.
    //----------------------------------------
    const std::string& getDataTypeName() const;

    //========================================
    //! \brief Get the version of the datatype contained in the payload after this header.
    //! \return Version major, minor of data type.
    //----------------------------------------
    const IcdVersion& getDataTypeVersion() const;

    //========================================
    //! \brief Get the original topic name of the datatype contained in the payload.
    //! \note Please note this method is slow because it is searching for the entry and copies the string.
    //! \return Topic name string.
    //----------------------------------------
    const std::string getTopicName() const;

    //========================================
    //! \brief Get the measurement timestamp of the datatype contained in the payload.
    //! \note Please note this method is slow because it is searching for the entry and copies the string.
    //! \return Measurement timestamp ntp time.
    //----------------------------------------
    const NtpTime getTimeStamp() const;

    //========================================
    //! \brief Get the list of optional additional entries in the header.
    //! These entries are unknown by the SDK and just passed through.
    //! \return List of additional entries.
    //----------------------------------------
    const EntryList& getAdditionalEntries() const;

    //========================================
    //! \brief Get the size of the content datatype payload following after this header.
    //! \return Size in bytes.
    //----------------------------------------
    std::size_t getPayloadSize() const;

public:
    //========================================
    //! \brief Read an icd header from icd stream.
    //! \param[in, out] is  Stream to read from.
    //! \return Either \c true if successful or \c false if not.
    //----------------------------------------
    bool deserialize(std::istream& is);

    //========================================
    //! \brief Write an icd header to icd stream.
    //! \param[in, out] os  Stream to write into.
    //! \return Either \c true if successful or \c false if not.
    //----------------------------------------
    bool serialize(std::ostream& os) const;

    //========================================
    //! \brief Calculate the size of the serialized header.
    //! \return Size in bytes.
    //! \note This might not be the same size as the header had when read with the readLE method!
    //----------------------------------------
    std::size_t getSerializedSize() const;

public:
    //========================================
    //! \brief Method to calculate a custom data container uuid for packed ICD datatypes.
    //! \returns The uuid matching to the given ICD datatype described by this header.
    //----------------------------------------
    DataContainerBase::Uuid computeUuid() const;

private:
    std::string m_dataTypeName; //!< Name of data type.
    IcdVersion m_dataTypeVersion; //!< Version of data type.
    std::size_t m_payloadSize; //! Size of payload/data type.

    EntryList m_additionalEntries; //!< List of additional unhandled/unknown entries.
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
