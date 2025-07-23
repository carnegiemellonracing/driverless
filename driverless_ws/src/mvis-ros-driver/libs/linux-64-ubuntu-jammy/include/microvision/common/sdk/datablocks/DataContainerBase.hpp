//==============================================================================
//! \file
//!
//!  Base class for all data containers
//!
//!  this includes all general data containers
//!  as well as all "old" type separated containers
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Sep 13, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>

#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid_io.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Base class for general idc data containers.
//!
//! Contains definitions of identification types for data containers.
//------------------------------------------------------------------------------
class DataContainerBase
{
    friend class ImporterBase;

public: //type declarations
    using HashId = uint64_t;

    using Uuid = boost::uuids::uuid;

    using Key = std::pair<DataTypeId, HashId>;

    class IdentificationKey;

public:
    //========================================
    //! \brief Default Constructor
    //----------------------------------------
    DataContainerBase() = default;

    //========================================
    //! \brief Constructor from idc data type used for serialization.
    //!
    //! \param[in] srcType  DataType
    //----------------------------------------
    explicit DataContainerBase(const DataTypeId::DataType srcType) : m_importedDataType(srcType) {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~DataContainerBase() = default;

public:
    //========================================
    //! \brief Return imported type of data type of which data was
    //!        transferred into general data.
    //! \return The imported serialized data type.
    //!
    //! \note This field is only set when imported by an idc device.
    //----------------------------------------
    DataTypeId::DataType getImportedType() const { return m_importedDataType; }

public:
    virtual uint64_t getClassIdHash() const = 0;

protected:
    //========================================
    //! \brief Set the original (serialized) data type id from which this data container has been deserialized.
    //!
    //! This is called by idc data container file and devices contained in the sdk. Plugin devices or other data sources should also set an id if possible.
    //!
    //! \param[in] dataType  Source data type.
    //----------------------------------------
    void setImportedDataType(const DataTypeId::DataType importedDataType) { m_importedDataType = importedDataType; }

protected:
    DataTypeId::DataType m_importedDataType{
        DataTypeId::DataType_Unknown}; //!< The data type id of the serialization this container has been filled with.
}; // DataContainerBase

//==============================================================================
//! \brief Key containing type/hash/uuid making an idc data container and custom data container uniquely identifiable.
//------------------------------------------------------------------------------
class DataContainerBase::IdentificationKey
{
public:
    //========================================
    //! \brief Removed default constructor - An identification key without ids does not make sense.
    //----------------------------------------
    IdentificationKey() = delete;

    //========================================
    //! \brief Create a data container type registration key.
    //!
    //! \param[in] id    idc datatype id
    //! \param[in] hash  Hash value of id string.
    //! \param[in] uuid  Global unique id for the type if it is a custom data container type.
    //----------------------------------------
    IdentificationKey(const DataTypeId id, const HashId hash, const Uuid uuid) : m_id{id}, m_hash{hash}, m_uuid{uuid} {}

    //========================================
    //! \brief Create a data container type registration key.
    //!
    //! \param[in] id    idc datatype id
    //! \param[in] hash  Hash value of id string.
    //!
    //! \note This constructor is not usable for custom data container identification.
    //----------------------------------------
    IdentificationKey(const DataTypeId id, const HashId hash) : m_id{id}, m_hash{hash}, m_uuid{Uuid()} {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~IdentificationKey() = default;

    friend bool operator==(const IdentificationKey& lhs, const IdentificationKey& rhs);
    friend bool operator!=(const IdentificationKey& lhs, const IdentificationKey& rhs);

public:
    //========================================
    //! \brief Get the idc datatype part of this identification key.
    //!
    //! \return The idc dataTypeId for this containers serialization.
    //----------------------------------------
    DataTypeId getId() const { return m_id; }

    //========================================
    //! \brief Get the hash part of this identification key.
    //!
    //! \return The hash value of this containers class hash id.
    //----------------------------------------
    HashId getHash() const { return m_hash; }

    //========================================
    //! \brief Get the custom data container unique id.
    //!
    //! \return The custom data container uuid/guid of this container. Will be nil for standard containers.
    //----------------------------------------
    Uuid getUuid() const { return m_uuid; }

private:
    DataTypeId m_id;
    HashId m_hash;
    Uuid m_uuid;
};

//========================================
//! \brief Nullable DataContainerBase pointer.
//----------------------------------------
using DataContainerPtr = std::shared_ptr<DataContainerBase>;

//==============================================================================

//==============================================================================
//! Compare with another key
//!
//! \param[in] lhs  First datatype identification key
//! \param[in] rhs  Second datatype identification key
//! \return True if equal, false if not.
//------------------------------------------------------------------------------
inline bool operator==(const DataContainerBase::IdentificationKey& lhs, const DataContainerBase::IdentificationKey& rhs)
{
    return (lhs.m_id == rhs.m_id) && (lhs.m_hash == rhs.m_hash) && (lhs.m_uuid == rhs.m_uuid);
}

//==============================================================================
//! Check with another key for inequality
//!
//! \param[in] lhs  First datatype identification key
//! \param[in] rhs  Second datatype identification key
//! \return True if not equal, false if equal.
//------------------------------------------------------------------------------
inline bool operator!=(const DataContainerBase::IdentificationKey& lhs, const DataContainerBase::IdentificationKey& rhs)
{
    return !(lhs == rhs);
}

ALLOW_WARNINGS_BEGIN
ALLOW_WARNING_DEPRECATED

//==============================================================================
//! \brief hash value calculation for a datatype registration key
//!
//! \param[in] key  Key to calculate the hash for
//! \return Calculated hash for the given key.
//!
//! Used for storing these keys in maps.
//------------------------------------------------------------------------------
inline std::size_t hash_value(const microvision::common::sdk::DataContainerBase::IdentificationKey& key)
{
    std::size_t seed = 0;
    boost::hash_combine(seed, key.getId());
    boost::hash_combine(seed, key.getHash());
    boost::hash_combine(seed, key.getUuid());
    return seed;
}

ALLOW_WARNINGS_END

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! \brief Base class for special idc data containers.
//------------------------------------------------------------------------------
class SpecializedDataContainer : public DataContainerBase
{
public:
    //========================================
    //! Default constructor
    //----------------------------------------
    SpecializedDataContainer() = default;

    //========================================
    //! Constructor from idc datatype
    //! \param[in] srcType  DataType
    //----------------------------------------
    explicit SpecializedDataContainer(const DataTypeId::DataType srcType) : DataContainerBase(srcType) {}

    //========================================
    //! Default destructor
    //----------------------------------------
    ~SpecializedDataContainer() override = default;

public: // getter
    // ========================================
    //! \brief Get the header device id.
    //! \return The header device id.
    //----------------------------------------
    virtual uint8_t getHeaderDeviceId() const { return m_headerDeviceId; }

    //========================================
    //! \brief get the header timestamp.
    //! \return The header timestamp.
    //----------------------------------------
    virtual NtpTime getHeaderNtpTime() const { return m_headerNtpTime; }

public:
    //========================================
    //! \brief Set the header data.
    //! \param[in] dh  IdcDataHeader which contains the
    //!                data to set the m_headerDeviceId
    //!                and m_headerNtpTime of this
    //!                DataContainer.
    //----------------------------------------
    void setDataHeader(const IdcDataHeader& dh)
    {
        m_headerDeviceId = dh.getDeviceId();
        m_headerNtpTime  = dh.getTimestamp();
    }

protected:
    uint8_t m_headerDeviceId{0}; //! Device id of the header.

    NtpTime m_headerNtpTime{}; //! Timestamp of the header.
};

//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
//! Custom data container class type finder helper construct used internally
//! to get a valid uuid/guid for ANY idc data container.
//!
//! \tparam T Class of which the uuid is requested.
//!
//! \note Only used internally for registering datatype Importers and Listeners!
//------------------------------------------------------------------------------
template<typename T>
struct GetUuidStaticFinder
{
    // check if class has getuuid function (which only is the case for CustomDataContainerBase derived classes)
    template<typename R>
    static std::true_type test(decltype(&R::getUuidStatic), int)
    {
        return std::true_type();
    }

    // fallback false (other idc data containers)
    template<typename R>
    static std::false_type test(...)
    {
        return std::false_type();
    }

    using Type = decltype(test<T>(nullptr, 0)); // evaluate test if member getuuid exists

    static const bool value = Type::value; // this tells after evaluation if getuuid exists or not!

    // if getuuid exists call this one (type is true then)
    static DataContainerBase::Uuid call(std::true_type) { return T::getUuidStatic(); }

    // fallback
    static DataContainerBase::Uuid call(...) { return DataContainerBase::Uuid(); }

    // the real call
    static DataContainerBase::Uuid getUuid() { return call(Type()); }
};

//==============================================================================
//! \brief Function to look up the uuid/guid of a data container class.
//!
//! Calls static method getUuidStatic.
//! Works even if the class does not have that method.
//!
//! \tparam T Class of which the uuid is requested.
//!
//! \note Only used internally for registering datatype Importers and Listeners!
//------------------------------------------------------------------------------
template<class T>
DataContainerBase::Uuid findUuid()
{
    return GetUuidStaticFinder<T>::getUuid();
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
namespace std {
//==============================================================================

//==============================================================================
//! \brief The template specializations of std::hash for \c IdentificationKey.
//------------------------------------------------------------------------------
template<>
struct hash<microvision::common::sdk::DataContainerBase::IdentificationKey>
{
    //========================================
    //! \brief Create hash of \c IdentificationKey.
    //----------------------------------------
    std::size_t operator()(microvision::common::sdk::DataContainerBase::IdentificationKey const& key) const
    {
        return hash_value(key);
    }
}; // :hash<IdentificationKey>

//==============================================================================
} // namespace std
//==============================================================================