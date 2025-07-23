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
//! \date May 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>
#include <microvision/common/sdk/datablocks/logpolygonlist2d/Polygon2d.hpp>
#include <microvision/common/sdk/Math.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//! \brief  A line in the two-dimensional plane, composed out of two points.
//! This class is a line composed out of two points and offers some
//! helper methods for simple geometrical calculations.
//!
//! \sa Vector2<float>, Polygon2dFloat
//------------------------------------------------------------------------------
template<typename T>
class Line2d final
{
public:
    enum class IntersectingType : uint8_t
    {
        NotIntersecting, ///< The lines are not intersecting, i.e. they are parallel or zero
        LineIntersecting, ///< The lines are intersecting within their line segments
        OutsideIntersecting ///< The lines are intersecting, but outside of their line segments
    };

public:
    using ValueType = T;

public:
    //========================================
    //! \brief Empty constructor
    //----------------------------------------
    Line2d() : first(), second() {}

    //========================================
    //! \brief Constructor with two points
    //----------------------------------------
    Line2d(const Vector2<ValueType>& a, const Vector2<ValueType>& b) : first(a), second(b) {}

    //========================================
    //! \brief Constructor with x/y coordinates of the two points given explicitly
    //----------------------------------------
    Line2d(ValueType x1, ValueType y1, ValueType x2, ValueType y2) : first(x1, y1), second(x2, y2) {}

    //========================================
    //! \brief  Constructor from a std::pair
    //----------------------------------------
    Line2d(const std::pair<Vector2<ValueType>, Vector2<ValueType>>& p) : first(p.first), second(p.second) {}

public:
    //========================================
    //! \name Accessor methods for information about this line
    //----------------------------------------
    //\{

    //========================================
    //! \brief Returns the first point
    //----------------------------------------
    const Vector2<ValueType>& getP1() const { return first; }

    //========================================
    //! \brief Returns the second point
    //----------------------------------------
    const Vector2<ValueType>& getP2() const { return second; }

    //========================================
    //! \brief Returns the length of this line, i.e. the distance of the two points.
    //----------------------------------------
    ValueType length() const { return getDiff().getLength(); }

    //========================================
    //! \brief Returns the difference between line end and line start as a Vector2<float>.
    //!
    //! The returned value is getP2() - getP1().
    //----------------------------------------
    Vector2<ValueType> getDiff() const { return second - first; }

    //========================================
    //! \brief Checks if the line is vertical (both x components are 0)
    //----------------------------------------
    bool isVertical() const { return std::abs(this->second.getX() - this->first.getX()) < 1E-11; }

    //========================================
    //! \brief Returns true if both points are zero
    //----------------------------------------
    bool isZero() const { return first.isZero() && second.isZero(); }

    //========================================
    //! \brief Returns the normal direction (orthogonal vector) to this line.
    //!
    //! The returned direction is the 90 degree rotated unit vector of the line.
    //----------------------------------------
    Vector2<ValueType> getNormalDirection() const
    {
        Vector2<float> diff = getDiff().getNormalized();
        return Vector2<float>(-diff.getY(), diff.getX());
    }

    //\}

public:
    //========================================
    //! \name Setter methods for changing this line
    //----------------------------------------
    //\{

    //========================================
    //! \brief Sets the first point.
    //----------------------------------------
    void setP1(const Vector2<ValueType>& p1) { first = p1; }

    //========================================
    //! \brief Sets the second point.
    //----------------------------------------
    void setP2(const Vector2<ValueType>& p2) { second = p2; }

    //\}

public:
    //========================================
    //! \name Geometrical relations to other objects.
    //----------------------------------------
    //\{

    //========================================
    //! \brief Calculates the intersection point between two lines.
    //! The returned value describes whether the two lines intersect
    //! within their line segments, or outside of their line segments,
    //! or not at all. The actual intersection point is written to
    //! intersectingPoint if that pointer is not a nullpointer.
    //----------------------------------------
    IntersectingType isIntersecting(const Line2d& other, Vector2<ValueType>* intersectingPoint = nullptr) const
    {
        if (this->isZero() || other.isZero())
        {
            return IntersectingType::NotIntersecting;
        }
        const bool has_inside_intersection = line2d_intersect(*this, other);

        IntersectingType result
            = has_inside_intersection ? IntersectingType::LineIntersecting : IntersectingType::OutsideIntersecting;
        Vector2<float> intersectPoint;

        Vector2<float> V = this->getDiff().getNormalized();
        Vector2<float> N = other.getNormalDirection();

        // check parallel lines
        // TODO no check if lanes are overlapping, but NotIntersecting is
        // probably correct
        if (std::abs(V * N) < 1E-11F)
        {
            result = IntersectingType::NotIntersecting;
        }
        else
        {
            float d = N * other.first;
            float u = (d - first.getX() * N.getX() - first.getY() * N.getY()) / (V * N);

            intersectPoint = first + V * u;
        }

        if (intersectingPoint)
        {
            *intersectingPoint = intersectPoint;
        }
        return result;
    }

    //========================================
    //! \brief Calculates and returns the point that results when projecting
    //!        the given point onto this line orthogonally.
    //----------------------------------------
    Vector2<ValueType> projectOntoLine(const Vector2<ValueType>& point) const
    {
        const Vector2<float> line_vec = getDiff();
        const float line_length       = length();
        if (std::abs(line_length) >= 1E-11F)
        {
            const Vector2<float> line_unitvec = line_vec / line_length;
            const Vector2<float> vec_to_point = point - first;
            const float length_of_projection  = line_unitvec * vec_to_point;
            return first + line_unitvec * length_of_projection;
        }
        else
        {
            // Zero-length line; what should we do?
            return point;
        }
    }

    //========================================
    //! \brief Returns true if this line "contains" the given point.
    //!
    //! Computes the distance from the point to the line, and if it is
    //! zero, the point lies on the line and this method returns true.
    //----------------------------------------
    bool containsPoint(const Vector2<ValueType>& point) const
    {
        float d = distanceFromLineSegment(point);
        return d < 1E-11F;
    }

    //========================================
    //! \brief Returns the distance of a point to this line segment.
    //!
    //! The distance of a point to a line segment is calculated and returned.
    //! If the point's projection onto the line is between the end
    //! points of this line, the distance to the projected point is
    //! returned. If the projection is not on the line segment, the
    //! distance to the closest end point is returned.
    //----------------------------------------

    ValueType distanceFromLineSegment(const Vector2<ValueType>& point) const
    {
        // coefficients for dot product and line length in order to get projDistanceFromP1
        const ValueType distanceXP1ToPoint = point.getX() - getP1().getX();
        const ValueType distanceYP1ToPoint = point.getY() - getP1().getY();
        const ValueType distanceXP1ToP2    = getP2().getX() - getP1().getX();
        const ValueType distanceYP1ToP2    = getP2().getY() - getP1().getY();

        const ValueType dot               = distanceXP1ToPoint * distanceXP1ToP2 + distanceYP1ToPoint * distanceYP1ToP2;
        const ValueType lineLengthSquared = distanceXP1ToP2 * distanceXP1ToP2 + distanceYP1ToP2 * distanceYP1ToP2;

        // projection distance from P1 - the fraction of the line that the point is closest to
        // projDistanceFromP1 multiplied with c respectively d gives the position (x,y) on the line closest to the point
        // if line has zero length, use -1
        const ValueType projDistanceFromP1
            = fuzzyCompareT<6>(lineLengthSquared, static_cast<ValueType>(0.0)) ? -1 : (dot / lineLengthSquared);

        ValueType closestPointOnLineSegmentX{};
        ValueType closestPointOnLineSegmentY{};

        // closest point in segment is P1
        if (projDistanceFromP1 < 0)
        {
            closestPointOnLineSegmentX = getP1().getX();
            closestPointOnLineSegmentY = getP1().getY();
        }
        // closest point in segment is P2
        else if (projDistanceFromP1 > 1)
        {
            closestPointOnLineSegmentX = getP2().getX();
            closestPointOnLineSegmentY = getP2().getY();
        }
        // closest point in segment is on line
        else
        {
            closestPointOnLineSegmentX = getP1().getX() + projDistanceFromP1 * distanceXP1ToP2;
            closestPointOnLineSegmentY = getP1().getY() + projDistanceFromP1 * distanceYP1ToP2;
        }

        const ValueType distanceX = point.getX() - closestPointOnLineSegmentX;
        const ValueType distanceY = point.getY() - closestPointOnLineSegmentY;

        // Distance between two points derived from Pythagorean theorem
        return static_cast<ValueType>(std::sqrt((distanceX * distanceX) + (distanceY * distanceY)));
    }

private:
    bool line2d_intersect(const Line2d& firstLine, const Line2d& secondLine) const
    {
        const float x1 = firstLine.first.getX();
        const float y1 = firstLine.first.getY();
        const float x2 = firstLine.second.getX();
        const float y2 = firstLine.second.getY();
        const float x3 = secondLine.first.getX();
        const float y3 = secondLine.first.getY();
        const float x4 = secondLine.second.getX();
        const float y4 = secondLine.second.getY();

        // a1, a2, b1, b2, c1, c2; // Coefficients of line eqns.
        // r1, r2, r3, r4;         // 'Sign' values

        const float a1 = y2 - y1;
        const float b1 = x1 - x2;
        const float c1 = (x2 * y1) - (x1 * y2);

        const float r3 = (a1 * x3) + (b1 * y3) + c1; // step 1
        const float r4 = (a1 * x4) + (b1 * y4) + c1; // step 2

        // step 3
        if (fuzzyFloatUnequalT<12>(r3, 0.0F) && fuzzyFloatUnequalT<12>(r4, 0.0F) && ((r3 * r4) >= 0))
        {
            return false;
        }
        const float a2 = y4 - y3;
        const float b2 = x3 - x4;
        const float c2 = (x4 * y3) - (x3 * y4);

        const float r1 = (a2 * x1) + (b2 * y1) + c2; // step 4
        const float r2 = (a2 * x2) + (b2 * y2) + c2; // step 5

        // step 6
        if (fuzzyFloatUnequalT<12>(r1, 0.0F) && fuzzyFloatUnequalT<12>(r2, 0.0F) && ((r1 * r2) >= 0.0F))
        {
            return false;
        }

        return true;
    }

private:
    //========================================
    //! \brief The first point of this line.
    //----------------------------------------
    Vector2<ValueType> first;

    //========================================
    //! \brief The second point of this line.
    //----------------------------------------
    Vector2<ValueType> second;
}; // Line2d

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
