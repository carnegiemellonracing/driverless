//==============================================================================
//! \filee
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 19, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/lanemarking/special/LaneMarkingSegmentIn6901.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//! \brief All segments representing one lane marking line.
//!
//! \sa microvision::common::sdk::LaneMarkingList6901
//------------------------------------------------------------------------------
class LaneMarkingIn6901 final
{
public:
    //! A vector of each lane segment in this lane
    using LaneMarkingSegments = std::vector<lanes::LaneMarkingSegmentIn6901>;

public:
    //========================================
    //! \brief The valid colors of lane markings.
    //----------------------------------------
    enum class LaneMarkingColor : uint32_t
    { //values are RGBA
        NotSpecified = 0x00000000U,
        White        = 0xFFFFFFFFU,
        Yellow       = 0xFFFF00FFU,
        Red          = 0xFF0000FFU,
        Blue         = 0x0000FFFFU,
        Green        = 0x00FF00FFU
    };

    //========================================
    //! \brief The possible classification of lane markings
    //----------------------------------------
    enum class LaneMarkingClassification : uint32_t
    {
        NotSpecified = 0x00000000U,
        SolidLane    = 0x00000001U,
        DashedLane   = 0x00000002U
    };

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    LaneMarkingIn6901() = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LaneMarkingIn6901() = default;

public: //getter
    //========================================
    //! \brief Get the id of this lane marking.
    //! \return The id.
    //----------------------------------------
    uint32_t getId() const { return m_id; }

    //========================================
    //! \brief Get a const reference to the vector of segments of this lane marking.
    //! \return The vector of segments.
    //----------------------------------------
    const LaneMarkingSegments& getLaneMarkingSegments() const { return m_segments; }

    //========================================
    //! \brief Get a reference to the vector of segments of this lane marking.
    //! \return The vector of segments.
    //----------------------------------------
    LaneMarkingSegments& getLaneMarkingSegments() { return m_segments; }

    //========================================
    //! \brief Get the id width this lane marking.
    //! \return The with of the lane marking.
    //----------------------------------------
    float getWidth() const { return m_width; }

    //========================================
    //! \brief Get the width variance of this lane marking.
    //! \return The width variance.
    //----------------------------------------
    float getWidthVariance() const { return m_widthVariance; }

    //========================================
    //! \brief Get the color of this lane marking.
    //! \return The color.
    //----------------------------------------
    LaneMarkingColor getColor() const { return m_color; }

    //========================================
    //! \brief Get the existence confidence.
    //! \return The existence confidence.
    //----------------------------------------
    float getExistenceConfidence() const { return m_existenceConfidence; }

    //========================================
    //! \brief Get the classification.
    //! \return The classification.
    //----------------------------------------
    LaneMarkingClassification getClassification() const { return m_classification; }

    //========================================
    //! \brief Get the classification confidence.
    //! \return The classification confidence.
    //----------------------------------------
    float getClassificationConfidence() const { return m_classificationConfidence; }

public: //setter
    //========================================
    //! \brief Set the id of this lane marking.
    //! \\param[in] id  The id.
    //----------------------------------------
    void setId(const uint32_t& id) { m_id = id; }

    //========================================
    //! \brief Set a vector of segments of this lane marking.
    //! \param[in] segments  The vector of segments.
    //----------------------------------------
    void setLaneMarkingSegments(const LaneMarkingSegments& segments) { m_segments = segments; }

    //========================================
    //! \brief Set the id width this lane marking.
    //! \param[in] width  The with of the lane marking.
    //----------------------------------------
    void setWidth(const float& width) { m_width = width; }

    //========================================
    //! \brief Set the width variance of this lane marking.
    //! \param[in] variance  The width variance.
    //----------------------------------------
    void setWidthVariance(const float& variance) { m_widthVariance = variance; }

    //========================================
    //! \brief Set the color of this lane marking.
    //! \param[in] color  The color.
    //----------------------------------------
    void setColor(const LaneMarkingColor& color) { m_color = color; }

    //========================================
    //! \brief Set the existence confidence of this lane marking.
    //! \param[in] confidence  The confidence.
    //----------------------------------------
    void setExistenceConfidence(const float& confidence) { m_existenceConfidence = confidence; }

    //========================================
    //! \brief Set the classification of this lane marking.
    //! \param[in] classification  The classification.
    //----------------------------------------
    void setClassification(const LaneMarkingClassification& classification) { m_classification = classification; }

    //========================================
    //! \brief Set the classification confidence of this lane marking.
    //! \param[in] confidence  The confidence.
    //----------------------------------------
    void setClassificationConfidence(const float& confidence) { m_classificationConfidence = confidence; }

private:
    uint32_t m_id{0}; //!< The id of the lane marking.
    LaneMarkingSegments m_segments{}; //!< Vector containing the segments of the lane marking.
    float m_width{0}; //!< The width of this lane marking in m.
    float m_widthVariance{0}; //!< The variance of the width in m.
    LaneMarkingColor m_color{}; //!< The color of the lane marking.
    float m_existenceConfidence{0}; //!< The confidence that this lane marking exists.
    LaneMarkingClassification m_classification{
        LaneMarkingClassification::NotSpecified}; //!< The classification of this lane marking.
    float m_classificationConfidence{0}; //!< The confidence that the classification is correct.

}; // LaneMarkingIn6901

//==============================================================================

bool operator==(const LaneMarkingIn6901& lhs, const LaneMarkingIn6901& rhs);

bool operator!=(const LaneMarkingIn6901& lhs, const LaneMarkingIn6901& rhs);

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
