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
//! \date Okt 19, 2020
//!
//! RoadBoundaryIn6902 represents road boundaries on the outer side of each road direction
//! It has the same attributes as the inner-lane markings (LaneMarking6901) but a different
//! datatype name
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundarySegmentIn6902.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief All segments representing one road boundary line.
//!
//! \sa microvision::common::sdk::RoadBoundaryList6902
//------------------------------------------------------------------------------
class RoadBoundaryIn6902 final
{
public:
    //! A vector of all segments in this road boundary.
    using RoadBoundarySegments = std::vector<RoadBoundarySegmentIn6902>;

public:
    //========================================
    //! \brief The valid colors of road boundaries.
    //----------------------------------------
    enum class RoadBoundaryColor : uint32_t
    { //values are RGBA
        NotSpecified = 0x00000000U,
        White        = 0xFFFFFFFFU,
        Yellow       = 0xFFFF00FFU,
        Red          = 0xFF0000FFU,
        Blue         = 0x0000FFFFU,
        Green        = 0x00FF00FFU,
        Orange       = 0xFFA500FFU
    };

    //========================================
    //! \brief The possible classification of road boundaries.
    //----------------------------------------
    enum class RoadBoundaryClassification : uint32_t
    {
        NotSpecified = 0x00000000U,
        Solid        = 0x00000001U,
        Dashed       = 0x00000002U,
        OffsetDotted = 0x00000003U,
        SolidDashed  = 0x00000004U

    };

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    RoadBoundaryIn6902() = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~RoadBoundaryIn6902() = default;

public: //getter
    //========================================
    //! \brief Get the id of this road boundary.
    //! \return The id.
    //----------------------------------------
    uint32_t getId() const { return m_id; }

    //========================================
    //! \brief Get a vector of segments of this road boundary.
    //! \return The vector of segments.
    //----------------------------------------
    const RoadBoundarySegments& getRoadBoundarySegments() const { return m_segments; }

    //========================================
    //! \brief Get the id width this road boundary.
    //! \return The with of the road boundary.
    //----------------------------------------
    float getWidth() const { return m_width; }

    //========================================
    //! \brief Get the width variance of this road boundary.
    //! \return The width variance.
    //----------------------------------------
    float getWidthVariance() const { return m_widthVariance; }

    //========================================
    //! \brief Get the color of this road boundary.
    //! \return The color.
    //----------------------------------------
    RoadBoundaryColor getColor() const { return m_color; }

    //========================================
    //! \brief Get the existence confidence.
    //! \return The existence confidence.
    //----------------------------------------
    float getExistenceConfidence() const { return m_existenceConfidence; }

    //========================================
    //! \brief Get the classification.
    //! \return The classification.
    //----------------------------------------
    RoadBoundaryClassification getClassification() const { return m_classification; }

    //========================================
    //! \brief Get the classification confidence.
    //! \return The classification confidence.
    //----------------------------------------
    float getClassificationConfidence() const { return m_classificationConfidence; }

public: //setter
    //========================================
    //! \brief Set the id of this road boundary.
    //! \\param[in] id  The id.
    //----------------------------------------
    void setId(const uint32_t& id) { m_id = id; }

    //========================================
    //! \brief Set a vector of segments of this road boundary.
    //! \param[in] segments  The vector of segments.
    //----------------------------------------
    void setRoadBoundarySegments(const RoadBoundarySegments& segments) { m_segments = segments; }

    //========================================
    //! \brief Set the id width this road boundary.
    //! \param[in] width  The with of the road boundary.
    //----------------------------------------
    void setWidth(const float& width) { m_width = width; }

    //========================================
    //! \brief Set the width variance of this road boundary.
    //! \param[in] variance  The width variance.
    //----------------------------------------
    void setWidthVariance(const float& variance) { m_widthVariance = variance; }

    //========================================
    //! \brief Set the color of this road boundary.
    //! \param[in] color  The color.
    //----------------------------------------
    void setColor(const RoadBoundaryColor& color) { m_color = color; }

    //========================================
    //! \brief Set the existence confidence of this road boundary.
    //! \param[in] confidence  The confidence.
    //----------------------------------------
    void setExistenceConfidence(const float& confidence) { m_existenceConfidence = confidence; }

    //========================================
    //! \brief Set the classification of this road boundary.
    //! \param[in] classification  The classification.
    //----------------------------------------
    void setClassification(const RoadBoundaryClassification& classification) { m_classification = classification; }

    //========================================
    //! \brief Set the classification confidence of this road boundary.
    //! \param[in] confidence  The confidence.
    //----------------------------------------
    void setClassificationConfidence(const float& confidence) { m_classificationConfidence = confidence; }

private:
    uint32_t m_id{0}; //!< The id of the road boundary.
    RoadBoundarySegments m_segments{}; //!< Vector containing the segments of the road boundary.
    float m_width{0}; //!< The width of this road boundary in m.
    float m_widthVariance{0}; //!< The variance of the width in m.
    RoadBoundaryColor m_color{}; //!< The color of the road boundary.
    float m_existenceConfidence{0}; //!< The confidence that this road boundary exists.
    RoadBoundaryClassification m_classification{
        RoadBoundaryClassification::NotSpecified}; //!< The classification of this road boundary.
    float m_classificationConfidence{0}; //!< The confidence that the classification is correct.

}; // RoadBoundaryIn6902

//==============================================================================

bool operator==(const RoadBoundaryIn6902& lhs, const RoadBoundaryIn6902& rhs);

bool operator!=(const RoadBoundaryIn6902& lhs, const RoadBoundaryIn6902& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
