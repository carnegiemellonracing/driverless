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
//! \date Aug 1, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/DataPackage.hpp>
#include <microvision/common/sdk/misc/Uri.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief Data package header for network payload.
//------------------------------------------------------------------------------
class NetworkDataPackage : public DataPackage
{
public:
    //========================================
    //! \brief Constructor.
    //!
    //! The index represents the order over packages like an index in a package stream.
    //! In case of a idc file it is the file cursor, otherwise it could be an iterator or
    //! something else depending to the source.
    //!
    //! \param[in] index        Index.
    //! \param[in] source       Source Uri.
    //! \param[in] destination  Destination Uri.
    //! \param[in] payload      Payload.
    //----------------------------------------
    NetworkDataPackage(const int64_t index        = 0,
                       const Uri& source          = Uri{},
                       const Uri& destination     = Uri{},
                       const PayloadType& payload = PayloadType{})
      : DataPackage{index, source, payload}, m_destination{destination}
    {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~NetworkDataPackage() override = default;

public:
    //========================================
    //! \brief Compare data packages for equality.
    //! \param[in] lhs  NetworkDataPackage to compare.
    //! \param[in] rhs  NetworkDataPackage to compare.
    //! \returns Either \c true if equals or otherwise \c false.
    //! \note This compares only the payload.
    //----------------------------------------
    friend bool operator==(const NetworkDataPackage& lhs, const NetworkDataPackage& rhs)
    {
        return (lhs.getPayload() == rhs.getPayload());
    }

    //========================================
    //! \brief Compare data packages for inequality.
    //! \param[in] lhs  NetworkDataPackage to compare.
    //! \param[in] rhs  NetworkDataPackage to compare.
    //! \returns Either \c true if unequals or otherwise \c false.
    //! \note This compares only the payload.
    //----------------------------------------
    friend bool operator!=(const NetworkDataPackage& lhs, const NetworkDataPackage& rhs)
    {
        return (lhs.getPayload() != rhs.getPayload());
    }

public: //getter
    //========================================
    //! \brief Get destination Uri.
    //! \return destination Uri.
    //----------------------------------------
    const Uri& getDestination() const { return m_destination; }

    //========================================
    //! \brief Get destination Uri.
    //! \return destination Uri.
    //----------------------------------------
    Uri& getDestination() { return m_destination; }

public: //setter
    //========================================
    //! \brief Set destination Uri.
    //! \param[in] destination  Destination Uri.
    //----------------------------------------
    void setDestination(const Uri& destination) { m_destination = destination; }

private:
    //========================================
    //! \brief Destination Uri.
    //----------------------------------------
    Uri m_destination;
};

//==============================================================================

//========================================
//! \brief Nullable NetworkDataPackage pointer.
//----------------------------------------
using NetworkDataPackagePtr = std::shared_ptr<NetworkDataPackage>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
