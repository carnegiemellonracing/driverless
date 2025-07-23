//==============================================================================
//! \file
//!
//! \brief Data package to store aggregated ldmi frame coming from ECU or MOVIA sensor.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jul 18th, 2024
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/SharedBuffer.hpp>

#include <array>

#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawStaticInfoIn2354.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data package to store reassembled ldmi aggregated frame.
//------------------------------------------------------------------------------
class LdmiAggregatedFrame2355 final : public SpecializedDataContainer
{
public:
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;

    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

private:
    //========================================
    //! \brief Logger name for payload setter.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::LdmiAggregatedFrame2355";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Size of aggregated frame.
    //!
    //! \note Complete message adds static message size.
    //----------------------------------------
    static constexpr std::size_t frameSize{0x2D76DU};

    //========================================
    //! \brief Size of static message.
    //!
    //! \note Complete message adds aggregated frame size.
    //----------------------------------------
    static constexpr std::size_t staticMessageSize{10308U};

    //========================================
    //! \brief Byte offset of reference id in static message A300 data.
    //----------------------------------------
    static constexpr std::size_t payloadReferenceIdOffset{192U};

    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    static constexpr const char* containerType{"sdk.specialcontainer.ldmiaggregatedframe2355"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    LdmiAggregatedFrame2355();

    //========================================
    //! \brief Construct instance by aggregated ldmi frame message payload and message payload of static message LdmiA300.
    //! \param[in] messagePayload     Aggregated ldmi frame message payload.
    //! \param[in] staticInfoPayload  LdmiA300 message payload.
    //----------------------------------------
    explicit LdmiAggregatedFrame2355(const SharedBuffer& messagePayload, const SharedBuffer& staticInfoPayload);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LdmiAggregatedFrame2355() override = default;

public:
    //========================================
    //! \brief Compares two ldmi raw frames for equality.
    //! \param[in] lhs  Ldmi aggregated frame 2355.
    //! \param[in] rhs  Ldmi aggregated frame 2355.
    //! \returns Either \c true if both frames are equal or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const LdmiAggregatedFrame2355& lhs, const LdmiAggregatedFrame2355& rhs);

    //========================================
    //! \brief Compares two ldmi aggregated frames for inequality.
    //! \param[in] lhs  Ldmi raw frame 2355.
    //! \param[in] rhs  Ldmi raw frame 2355.
    //! \returns Either \c true if both frames are unequal or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const LdmiAggregatedFrame2355& lhs, const LdmiAggregatedFrame2355& rhs);

public: // DataContainerBase implementation
    uint64_t getClassIdHash() const override;

public: // getter
    //========================================
    //! \brief Get ldmi raw static info.
    //! \returns Ldmi static info.
    //----------------------------------------
    const SharedBuffer& getStaticInfo() const;

    //========================================
    //! \brief Get ldmi static info reference id config hash.
    //! \returns Ldmi static info reference id config hash.
    //----------------------------------------
    const uint64_t& getStaticInfoReferenceIdConfigHash() const;

    //========================================
    //! \brief Get aggregated frame message payload.
    //!
    //! Serialized form of aggregated ldmi frame which contains all sensor data.
    //!
    //! \return Message payload.
    //----------------------------------------
    const SharedBuffer& getMessagePayload() const;

public: // setter
    //========================================
    //! \brief Set ldmi raw static info.
    //!
    //! Updates the reference id config hash.
    //!
    //! \param[in] config  Ldmi raw static info (A300 message).
    //----------------------------------------
    void setStaticInfo(const SharedBuffer& config);

    //========================================
    //! \brief Set aggregated frame payload.
    //! \param[in] messagePayload  Message payload.
    //----------------------------------------
    void setMessagePayload(const SharedBuffer& messagePayload);

private:
    //========================================
    //! \brief Ldmi static info (A300 message) reference id.
    //!
    //! Updated when the static info is set.
    //----------------------------------------
    uint64_t m_ReferenceId;

    //========================================
    //! \brief Ldmi static info (A300 message).
    //----------------------------------------
    SharedBuffer m_staticInfo;

    //========================================
    //! \brief Aggregated frame message payload.
    //----------------------------------------
    SharedBuffer m_messagePayload;
};

//==============================================================================
//! \brief Nullable LdmiAggregatedFrame2355 pointer.
//------------------------------------------------------------------------------
using LdmiAggregatedFrame2355Ptr = std::shared_ptr<LdmiAggregatedFrame2355>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
