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
//! \date Sep 12, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/metainformationlist/special/MetaInformationBaseIn7110.hpp>

#include <microvision/common/sdk/misc/crypto/Md5.h>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief
//!
//! This class holds a configuration file used during this recording.
//------------------------------------------------------------------------------
class MetaInformationConfigFileIn7110 : public MetaInformationBaseIn7110
{
public:
    //========================================
    //! \brief Type source of this config file.
    //----------------------------------------
    enum class FileSource : uint16_t
    {
        Undefined               = 0,
        DeviceThirdPartySLms100 = 1,
        DeviceThirdPartySLms200 = 2,
        DeviceThirdPartySLms500 = 3
    }; // FileSource

    //========================================
    //! \brief Type of file.
    //----------------------------------------
    enum class FileType : uint16_t
    {
        Undefined = 0,
        Xml       = 1,
        Text      = 2,
        Binary    = 3
    }; // FileType

    //========================================
    //! \brief Type of checksum for config files.
    //----------------------------------------
    using Md5Checksum = std::array<uint8_t, crypto::Md5::digestLength>;

public:
    //========================================
    //! \brief Constructor.
    //!
    //! Initializes this instance as meta information of type config file.
    //----------------------------------------
    MetaInformationConfigFileIn7110()
      : MetaInformationBaseIn7110(MetaInformationBaseIn7110::MetaInformationType::ConfigFile)
    {}

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~MetaInformationConfigFileIn7110() override = default;

public: // getter
    //========================================
    //! \brief Get the source of this file.
    //!
    //! \return The source of this file.
    //----------------------------------------
    FileSource getFileSource() const { return m_fileSource; }

    //========================================
    //! \brief Get the file type.
    //!
    //! \return The file type.
    //----------------------------------------
    FileType getFileType() const { return m_fileType; }

    //========================================
    //! \brief Get the file name.
    //!
    //! \return The name of the file with extension but without directory.
    //----------------------------------------
    const std::string& getFileName() const { return m_fileName; }

    //========================================
    //! \brief Get the checksum.
    //!
    //! \return The MD5 hash of the file content.
    //----------------------------------------
    const Md5Checksum& getChecksum() const { return m_checksum; }

    //========================================
    //! \brief Get the file content.
    //!
    //! \return The file content as binary buffer.
    //----------------------------------------
    const std::vector<uint8_t>& getFileContent() const { return m_fileContent; }

    //========================================
    //! \brief Checks whether the file content is valid.
    //!
    //! \return \c True, if the content is valid (i.e. the checksum is correct), or \c false otherwise.
    //----------------------------------------
    bool isFileContentValid() const;

public: // setter
    //========================================
    //! \brief Set the source of this file.
    //!
    //! \param[in] fileSource  The new source of this file.
    //----------------------------------------
    void setFileSource(const FileSource fileSource) { m_fileSource = fileSource; }

    //========================================
    //! \brief Set the file type.
    //!
    //! \param[in] fileType  The new file type.
    //----------------------------------------
    void setFileType(const FileType fileType) { m_fileType = fileType; }

    //========================================
    //! \brief Set the file name.
    //!
    //! \param[in] fileName  The new name of the file with extension but without directory.
    //----------------------------------------
    void setFileName(const std::string& fileName) { m_fileName = fileName; }

    //========================================
    //! \brief Set the checksum.
    //!
    //! \param[in] checksum  The new checksum.
    //! See \a setFileContent for setting the checksum automatically.
    //----------------------------------------
    void setChecksum(const Md5Checksum& checksum) { m_checksum = checksum; }

    //========================================
    //! \brief Set the file content.
    //!
    //! \param[in] content               The new file content as binary buffer.
    //! \param[in] shouldUpdateChecksum  If set to \c true, the checksum will be calculated automatically and set into
    //!                                  the corresponding field, otherwise the checksum field will not be changed.
    //! See \a setChecksum for setting the checksum manually.
    //----------------------------------------
    void setFileContent(const std::vector<uint8_t>& content, bool shouldUpdateChecksum = true);

    //========================================
    //! \brief Calculate the checksum of the current file content and set it into the checksum field.
    //!
    //! See \a setChecksum for setting the checksum manually.
    //----------------------------------------
    void updateChecksum();

public:
    //========================================
    //! \brief Tests this meta information for equality.
    //!
    //! \param[in] otherBase  The other meta information to compare with.
    //! \return \c True, if the two meta information are equal, \c false otherwise.
    //----------------------------------------
    bool isEqual(const MetaInformationBaseIn7110& otherBase) const override;

    //========================================
    //! \brief Get the size of the serialized payload.
    //!
    //! \return The size of the serialized payload.
    //----------------------------------------
    uint32_t getSerializedPayloadSize() const override;

    //========================================
    //! \brief Deserialize the data from a stream.
    //!
    //! \param[in,out] is           The stream to read the data from.
    //! \param[in]     payloadSize  The size of the payload in the stream.
    //! \return \c True, if the deserialization was successful, \c false otherwise.
    //----------------------------------------
    bool deserializePayload(std::istream& is, const uint32_t payloadSize) override;

    //========================================
    //! \brief Serialize the data into a stream.
    //!
    //! \param[in,out] os           The stream to write the data to.
    //! \return \c True, if the serialization was successful, \c false otherwise.
    //----------------------------------------
    bool serializePayload(std::ostream& os) const override;

private:
    static Md5Checksum calculateChecksum(const std::vector<uint8_t>& buffer);

private:
    FileSource m_fileSource{FileSource::Undefined};
    FileType m_fileType{FileType::Undefined};
    std::string m_fileName{};
    Md5Checksum m_checksum{};
    std::vector<uint8_t> m_fileContent;
}; // MetaInformationSyncBoxConfigIn7110

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================
