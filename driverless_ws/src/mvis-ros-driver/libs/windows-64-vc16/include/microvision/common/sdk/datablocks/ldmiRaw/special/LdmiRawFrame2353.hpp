//==============================================================================
//! \file
//!
//! \brief Data package to store reassembled ldmi raw frame coming from MOVIA Lidar B1 sensor.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 28th, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawStaticInfoIn2353.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrameFooterIn2353.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrameHeaderIn2353.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrameRowIn2353.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Data package to store reassembled ldmi raw frame.
//------------------------------------------------------------------------------
class LdmiRawFrame2353 final : public SpecializedDataContainer
{
public:
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;

    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    static constexpr const char* containerType{"sdk.specialcontainer.ldmirawframe2353"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Array type of ldmi rows.
    //----------------------------------------
    using RowsType = std::array<LdmiRawFrameRowIn2353, LdmiRawFrameRowIn2353::maxNumberOfRows>;

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    LdmiRawFrame2353();

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LdmiRawFrame2353() override = default;

public:
    //========================================
    //! \brief Compares two ldmi raw frames for equality.
    //! \param[in] lhs  Ldmi raw frame 2353.
    //! \param[in] rhs  Ldmi raw frame 2353.
    //! \returns Either \c true if both frames are equal or otherwise \c false.
    //----------------------------------------
    friend bool operator==(const LdmiRawFrame2353& lhs, const LdmiRawFrame2353& rhs);

    //========================================
    //! \brief Compares two ldmi raw frames for inequality.
    //! \param[in] lhs  Ldmi raw frame 2353.
    //! \param[in] rhs  Ldmi raw frame 2353.
    //! \returns Either \c true if both frames are unequal or otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const LdmiRawFrame2353& lhs, const LdmiRawFrame2353& rhs);

public: // DataContainerBase implementation
    uint64_t getClassIdHash() const override;

public: // getter
    //========================================
    //! \brief Get ldmi raw static info.
    //! \returns Ldmi static info.
    //----------------------------------------
    const LdmiRawStaticInfoIn2353& getStaticInfo() const;

    //========================================
    //! \brief Get ldmi raw frame header.
    //! \returns Ldmi frame header.
    //----------------------------------------
    const LdmiRawFrameHeaderIn2353& getHeader() const;

    //========================================
    //! \brief Get ldmi raw frame rows.
    //! \returns Array of ldmi raw frame row.
    //----------------------------------------
    const RowsType& getRows() const;

    //========================================
    //! \brief Get size of stored ldmi raw frame rows.
    //! \returns Number of ldmi raw frame rows.
    //----------------------------------------
    std::size_t getRowSize() const;

    //========================================
    //! \brief Get ldmi raw frame footer.
    //! \returns Ldmi frame footer.
    //----------------------------------------
    const LdmiRawFrameFooterIn2353& getFooter() const;

public: // setter
    //========================================
    //! \brief Set ldmi raw static info.
    //! \param[in] config  Ldmi raw static info.
    //----------------------------------------
    void setStaticInfo(const LdmiRawStaticInfoIn2353& config);

    //========================================
    //! \brief Set ldmi raw frame header.
    //! \param[in] header  Ldmi raw frame header.
    //----------------------------------------
    void setHeader(const LdmiRawFrameHeaderIn2353& header);

    //========================================
    //! \brief Set ldmi raw frame row.
    //! \note Row will set by row id as index of array.
    //! \param[in] row  Ldmi raw frame row.
    //! \returns Either \c true if row is not set or out of range, otherwise \c
    //! false.
    //----------------------------------------
    bool setRow(const LdmiRawFrameRowIn2353& row);

    //========================================
    //! \brief Set ldmi raw frame footer package.
    //! \param[in] footer  Ldmi raw frame footer.
    //----------------------------------------
    void setFooter(const LdmiRawFrameFooterIn2353 footer);

public:
    //========================================
    //! \brief Check if the frame is complete.
    //!
    //! Frame is complete if the static configuration,
    //! frame header, frame rows and frame footer is collected.
    //!
    //! \returns Either \c true if frame is complete or otherwise \c false.
    //----------------------------------------
    bool isComplete() const;

    //========================================
    //! \brief Get the number of rows which is expected by configuration.
    //! \returns Number of rows which are expected.
    //----------------------------------------
    std::size_t getNumberOfRows() const;

private:
    //========================================
    //! \brief Ldmi static info.
    //----------------------------------------
    LdmiRawStaticInfoIn2353 m_staticInfo;

    //========================================
    //! \brief Ldmi frame header.
    //----------------------------------------
    LdmiRawFrameHeaderIn2353 m_header;

    //========================================
    //! \brief Ldmi frame rows.
    //----------------------------------------
    RowsType m_rows;

    //========================================
    //! \brief Ldmi frame rows collected.
    //----------------------------------------
    std::size_t m_rowsSize;

    //========================================
    //! \brief Ldmi frame footer.
    //----------------------------------------
    LdmiRawFrameFooterIn2353 m_footer;
};

//==============================================================================
//! \brief Nullable LdmiRawFrame2353 pointer.
//------------------------------------------------------------------------------
using LdmiRawFrame2353Ptr = std::shared_ptr<LdmiRawFrame2353>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
