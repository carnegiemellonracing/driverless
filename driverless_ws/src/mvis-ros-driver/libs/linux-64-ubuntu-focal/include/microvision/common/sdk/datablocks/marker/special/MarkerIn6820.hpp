//==============================================================================
//! \file
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Quaternion.hpp>
#include <microvision/common/sdk/Vector3.hpp>
#include <microvision/common/sdk/misc/RgbaColor.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Marker type used in MarkerList6820.
//!
//! Each marker represents a single geometric entity or and ensemble of
//! identical entities depending on the marker type.
//!
//! \note When assigning data to a C++ marker object, follow the order,
//!       -# marker type
//!       -# points/vertices
//!       -# colors
//! \note The minimal number of provided colors has to be one. If no color is
//!       provided, one color entry will be generated with #defaultColor.
//! \note If the number of colors provided is smaller than the number expected,
//!       the last provided color is assumed to be valid for the remaining number
//!       of items.
//! \note The marker and vertex positions are given in meter.
//------------------------------------------------------------------------------
class MarkerIn6820
{
public:
    //========================================
    //! \brief Type of this marker.
    //!
    //! \note The TextOnly marker type is part of all marker types.
    //----------------------------------------
    enum class MarkerType : uint8_t
    {
        //========================================
        //!\brief An arrow (1) between start and end point or (2)
        //!       from position with a given length.
        //!
        //! # If m_points is filled:
        //!   * start point: m_points[0]
        //!   * end point: m_points[1]
        //!   * shaft diameter : scale.x
        //!   * head diameter: scale.y
        //!   * head length: scale.z if $\neq 0$.
        //!   * color: colors[0].
        //! # If m_points is not filled
        //!   * start point: position
        //!   * orientation: around start point with identity is along
        //!                  the +x axis.
        //!   * arrow length: scale.x
        //!   * head diameter: scale.y
        //!   * head length: scale.z
        //!   * color: colors[0].
        //----------------------------------------
        Arrow = 0,

        //========================================
        //!\brief A cube.
        //!
        //! * position: center of the cube and pivot point.
        //! * orientation: identity is along the +x axis.
        //! * size: scale x, y and z in meter.
        //! * color: (Only one) color of the cube.
        //----------------------------------------
        Cube = 1,

        //========================================
        //!\brief A sphere.
        //!
        //! * position: center of the sphere and pivot point
        //! * radius: scale.x
        //! * color: (Only one) color of the shpere.
        //----------------------------------------
        Sphere = 2,

        //========================================
        //!\brief A cylinder with the height orthogonal to the xy plane.
        //!
        //! * position: center of the cylinder and pivot point
        //! * radius in x: scale.x
        //! * radius in y: scale.y
        //! * height: scale.z
        //! * color: (Only one) color of the cylinder.
        //----------------------------------------
        Cylinder = 3,

        //========================================
        //!\brief A line strip.
        //!
        //! * position and orientation will be used as
        //!   a transformation for the line points given in
        //!   m_points.
        //! * points: The line strip will be drawn between
        //!           two consecutive points in m_points,
        //!           i.e. 0-1, 1-2, 2-3, etc.
        //! * colors: (Only one) color for the whole line strip.
        //! * line diameter: scale.x [in pixels!!!]
        //----------------------------------------
        LineStrip = 4,

        //========================================
        //!\brief A list of lines
        //!
        //! * position and orientation will be used as
        //!   a transformation for the line points given in
        //!   m_points.
        //! * points: The line strip will be drawn between
        //!           every pair of points in m_points,
        //!           i.e. 0-1, 2-3, 4-5, etc.
        //! * colors: Per line color in m_color. While there
        //!           are n vertices, the number of colors
        //!           is limited to n/2.
        //! * line diameter: scale.x [in pixels!!!]
        //----------------------------------------
        LineList = 5,

        //========================================
        //!\brief A list of identical cubes at different positions.
        //!
        //! * position and orientation will be used as
        //!   a transformation for the center points given in
        //!   m_points.
        //! * points: center of the cubes.
        //! * scale: size
        //! * colors: Per cube color in m_color.
        //!
        //! \note: Cubes in a CubeList do not have individual orientation.
        //----------------------------------------
        CubeList = 6,

        //========================================
        //!\brief A list of identical spheres at different positions.
        //!
        //! * position and orientation will be used as
        //!   a transformation for the center points given in
        //!   m_points.
        //! * center of the sphere: points.
        //! * radius: scale.x
        //! * colors: Per cube color in m_color.
        //----------------------------------------
        SphereList = 7,

        //========================================
        //!\brief List of points.
        //!
        //! * position and orientation will be used as
        //!   a transformation for the points given in
        //!   m_points.
        //! * points: the position of the points.
        //! * point radius: scale.x [in pixels!!!]
        //! * colors: Per vertex color in m_color.
        //!
        //! \note: Not supporting separate width and height, only radius
        //!        for points.
        //----------------------------------------
        Points = 8,

        //========================================
        //!\brief Only the text will be shown.
        //!
        //! * position: The text will be shown at the given position.
        //! * colors: Color of the text.
        //----------------------------------------
        TextOnly = 9,

        //========================================
        //!\brief A list of (filled) triangles.
        //!
        //! * position and orientation will be used as
        //!   a transformation for the triangle points given in
        //!   m_points.
        //! * points: The filled triangles will be drawn between each
        //!           triple of points in m_points,
        //!           i.e. 0-1-2, 3-4-5, 6-7-8, etc.
        //! * line diameter: scale.x
        //! * colors: Per triangle one color in m_color.
        //----------------------------------------
        TriangleList = 11
    }; // MarkerType

    //========================================
    //! \brief A cartesian 3D position.
    //----------------------------------------
    using Position = Vector3<float>;

    //========================================
    //! \brief A 3D orientation.
    //----------------------------------------
    using Orientation = Quaternion<float>;

    //========================================
    //! \brief 3D scale factors.
    //----------------------------------------
    using Scale = Vector3<float>;

    //========================================
    //! \brief Color type used by MarkerIn6820.
    //----------------------------------------
    using Color = RgbaColorFloat;

    //========================================
    //! \brief A point in a 3D cartesian space.
    //----------------------------------------
    using Point = Vector3<float>;

    //========================================
    //! \brief A vector of points.
    //----------------------------------------
    using PointVector = std::vector<Point>;

    //========================================
    //! \brief A vector of colors.
    //----------------------------------------
    using ColorVector = std::vector<Color>;

    //========================================
    //! \brief A vector of strings used to define
    //!        a marker's namespace.
    //----------------------------------------
    using MarkerNamespace = std::vector<std::string>;

private:
    //========================================
    //! \brief Default color (white) used by MarkerIn6820.
    //----------------------------------------
    static const Color defaultColor;

    //========================================
    //! \brief The value of the flag to be set, if
    //!        the marker is using point/color vectors.
    //----------------------------------------
    static constexpr uint8_t usingVectorsFlag{0x01U};

    //========================================
    //! \brief The value of the flag to be set, if
    //!        the marker is coordinate system locked.
    //----------------------------------------
    static constexpr uint8_t isCoordinateSystemLockedFlag{0x02U};

    //========================================
    //! \brief The value of the flag to be set, if
    //!        the namespace given in the marker is
    //!        complete or will use the marker list
    //!        namespace prefix.
    //----------------------------------------
    static constexpr uint8_t isNamespaceCompleteFlag{0x04U};

public:
    //========================================
    //! \brief Get the default color.
    //! \return A const reference to the default color.
    //----------------------------------------
    static const Color& getDefaultColor();

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    MarkerIn6820();

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~MarkerIn6820() = default;

public:
    //========================================
    //! \brief Deserialize this MarkerIn620 from the
    //!        input stream \a is.
    //! \param[in,out] is  The input stream the marker
    //!                    shall be deserialized from.
    //! \return \c True if the deserialization was
    //!         sucessfull, \c false otherwise.
    //----------------------------------------
    bool deserialize(std::istream& is);

    //========================================
    //! \brief Serialize this MarkerIn620 to the
    //!        output stream \a os.
    //! \param[in,out] os  The output stream the marker
    //!                    shall be serialized to.
    //! \return \c True if the serialization was
    //!         sucessfull, \c false otherwise.
    //----------------------------------------
    bool serialize(std::ostream& os) const;

    //========================================
    //! \brief Get size in bytes of this MarkerList6820 as serialization.
    //! \return Size of the serialization in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize() const;

public:
    //========================================
    //! \brief Get the typer use for this marker.
    //! \return Return marker's type.
    //----------------------------------------
    MarkerType getMarkerType() const { return m_markerType; }

    //========================================
    //! \brief Depending on the marker type it contains
    //!        vertices or not.
    //! \return \c True if the marker type is using
    //!         the point array. \c false otherwise.
    //! \note MarkerType::Arrow is a special case, it
    //!       may or may not make use of the point
    //!       vector. Since it decides its behavior on
    //!       the size of the points vector, \c true
    //!       will be returned for MarkerType::Arrow.
    //!       If the point vector is not used, the
    //!       serialization of it will be completely skipped.
    //----------------------------------------
    bool isMarkerUsingVertices() const { return m_markerIsUsingVertices; }

    //========================================
    //! \brief Get whether the markers namespace
    //!        is complete.
    //! \return \c True if the namespace is complete,
    //!         \c false if the marker list's namespace
    //!         prefix is needed to complete the marker's
    //!         namespace.
    //----------------------------------------
    bool isMarkerNamespaceComplete() const { return m_isNamespaceComplete; }

    //========================================
    //! \brief Get the marker's namespace.
    //! \return Return the marker's namespace.
    //----------------------------------------
    const MarkerNamespace& getMarkerNamespace() const { return m_markerNamespace; }

    //========================================
    //! \brief Get the marker's namespace.
    //! \return Return the marker's namespace.
    //----------------------------------------
    MarkerNamespace& getMarkerNamespace() { return m_markerNamespace; }

    //========================================
    //! \brief Get the position use for this marker.
    //!
    //! Together with the orientation, position is
    //! defining a pose transformation for points
    //! in m_points, in case the marker is using them.
    //! \return Return marker's position.
    //----------------------------------------
    const Position& getPosition() const { return m_position; }

    //========================================
    //! \brief Get the orientation use for this marker.
    //!
    //! Together with the position, orientation is
    //! defining a pose transformation for points
    //! in m_points, in case the marker is using them.
    //! \return Return marker's orientation.
    //----------------------------------------
    const Orientation& getOrientation() const { return m_orientation; }

    //========================================
    //! \brief Get the scale values.
    //! \return Return the scale values.
    //----------------------------------------
    const Scale& getScale() const { return m_scale; }

    //========================================
    //! \brief Get the color used for a non-list
    //!        marker type.
    //!
    //! Internally the color is stored in m_colors
    //! as its first element.
    //!
    //! \return Get the color used for a non-list
    //!         marker type.
    //----------------------------------------
    const Color getColor() const;

    //========================================
    //! \brief Get the marker's point vector.
    //! \return A const reference to the point vector.
    //----------------------------------------
    const PointVector& getPointVector() const { return m_points; }

    //========================================
    //! \brief Get the marker's point vector.
    //! \return A reference to the point vector.
    //----------------------------------------
    PointVector movePointVector()
    {
        PointVector points = std::move(m_points);
        normalize(m_markerType, m_points, m_colors);
        return points;
    }

    //========================================
    //! \brief Get the marker's color vector.
    //! \return A const reference to the color vector.
    //----------------------------------------
    const ColorVector& getColorVector() const { return m_colors; }

    //========================================
    //! \brief Get the marker's color vector.
    //! \return A reference to the color vector.
    //----------------------------------------
    ColorVector moveColorVector()
    {
        ColorVector colors = std::move(m_colors);
        normalize(m_markerType, m_points, m_colors);
        return colors;
    }

    //========================================
    //! \brief Get the marker's text.
    //! \return Return the marker's text.
    //----------------------------------------
    const std::string& getText() const { return m_text; }

    //========================================
    //! \brief Get the marker's lifetime in milliseconds.
    //! \return Return the marker's lifetime in milliseconds.
    //----------------------------------------
    uint32_t getLifetimeInMilliseconds() const { return m_lifetimeInMilliseconds; }

    //========================================
    //! \brief Get whether the marker's position is locked to the
    //!        coordinate system while its lifetime.
    //!
    //! E.g. if the coordinate system of the marker is the vehicle pose,
    //! its position will be updated if the vehicle pose has changed.
    //!
    //! \return Whether the marker is coordinate system locked.
    //----------------------------------------
    bool isCoordinateSystemLockedWhileLifetime() const { return m_isCoordinateSystemLockedWhileLifetime; }

    //========================================
    //! \brief Get the marker ID.
    //! \return Return the marker ID.
    //----------------------------------------
    int32_t getMarkerId() const { return m_markerId; }

    //========================================
    //! \brief Get the data of the reserved part of this datatype.
    //! \return A const reference to the reserved data of this datatype.
    //! \note If this vector is not empty this means that the serialization
    //!       from which this marker has been read has a newer version than
    //!       this implementation. Reserved data are stored for forward
    //!       compatibility to be able to forward also data from a newer
    //!       version.
    //----------------------------------------
    const std::vector<char>& getReserved() const { return m_reserved; }

public:
    //========================================
    //! \brief Set the marker's type.
    //! \param[in] newType  The new marker's type.
    //! \attention Calling this method will also reset
    //!            the content of the points and color
    //!            vector if they are not used for this
    //!            marker type.
    //!            For an MarkerType::Arrow, the points vector will
    //!            be downsized to 2 if its size is larger
    //!            than 2.
    //----------------------------------------
    void setMarkerType(const MarkerType newType)
    {
        m_markerType            = newType;
        m_markerIsUsingVertices = markerTypeToWhetherUsingVertices();
        normalize(m_markerType, m_points, m_colors);
    }

    //========================================
    //! \brief Set the new marker's position.
    //! \param[in] newPosition  The new marker's position.
    //----------------------------------------
    void setPosition(const Position& newPosition) { m_position = newPosition; }

    //========================================
    //! \brief Set the new marker's orientation.
    //! \param[in] newOrientation  The new marker's orientation.
    //----------------------------------------
    void setOrientation(const Orientation& newOrientation) { m_orientation = newOrientation; }

    //========================================
    //! \brief Set the new marker's scale.
    //! \param[in] newScale  The new marker's scale factors.
    //----------------------------------------
    void setScale(const Scale& newScale) { m_scale = newScale; }

    //========================================
    //! \brief Set the color used for a non-list
    //!        marker type.
    //!
    //! Internally the color is stored in m_colors
    //! as its first element.
    //!
    //! \param[in] newColor  New color used for a
    //!                      non-list marker.
    //! \attention Using setColor will overwrite the
    //!            first color entry.
    //----------------------------------------
    void setColor(const Color& newColor);

    //========================================
    //! \brief Set a new point vector.
    //! \param[in] newPoints  The new points vector.
    //!                       The vector content will be moved.
    //! \return Number of points that have been accepted.
    //!         Depending on the marker's type, certain criteria
    //!         has to be fulfilled by the number of an non-empty
    //!         point vector. E.g. needs a MarkerType::LineStrip
    //!         at least 2 points, whereas a MarkerType::LineList
    //!         needs a multiple of 2 points.
    //! \note Not all marker types make use
    //!       of this entry
    //! \attention setPointVector has to be called after
    //!            setMarkerType to avoid loosing entries due
    //!            to internal resizing.
    //----------------------------------------
    uint32_t setPointVector(const PointVector& newPoints)
    {
        m_points = newPoints;
        normalize(m_markerType, m_points, m_colors);
        return static_cast<uint32_t>(m_points.size());
    }

    //========================================
    //! \brief Set a new point vector.
    //! \param[in, out] newPoints  The new points vector.
    //!                            The vector content will be moved.
    //! \return Number of points that have been accepted.
    //!         Depending on the marker's type, certain criteria
    //!         has to be fulfilled by the number of an non-empty
    //!         point vector. E.g. needs a MarkerType::LineStrip
    //!         at least 2 points, whereas a MarkerType::LineList
    //!         needs a multiple of 2 points.
    //! \note Not all marker types make use
    //!       of this entry
    //! \attention setPointVector has to be called after
    //!            setMarkerType to avoid loosing entries due
    //!            to internal resizing.
    //----------------------------------------
    uint32_t setPointVector(PointVector&& newPoints)
    {
        m_points = std::move(newPoints);
        normalize(m_markerType, m_points, m_colors);
        return static_cast<uint32_t>(m_points.size());
    }

    //========================================
    //! \brief Set the per point colors used for a list
    //!        marker type.
    //!
    //! \param[in] newColors  Vector of colors to be used
    //!                       for the marker.
    //! \attention setColorVector has to be called after
    //!            setPointVector and setMarkerType to
    //!            avoid loosing entries due to internal
    //!            resizing.
    //! \note Depending on the marker type the colors
    //!       are assigned indexwise to the points or
    //!       to the elements as lines or line strip.
    //! \note If too many colors are provided, the
    //!       color vector will be resized. If \a newColor
    //!       is empty, the color vector will set to
    //!       hold one element of the default color, to
    //!       the vector will always contain at least one
    //!       color.
    //! \attention Using setColor will overwrite the
    //!            first color entry.
    //----------------------------------------
    uint32_t setColorVector(const ColorVector& newColors)
    {
        m_colors = newColors;
        normalize(m_markerType, m_points, m_colors);
        return static_cast<uint32_t>(m_points.size());
    }

    //========================================
    //! \brief Set the per point colors used for a list
    //!        marker type.
    //!
    //! \param[in,out] newColors  Vector of colors to be used
    //!                           for the marker.
    //! \attention setColorVector has to be called after
    //!            setPointVector and setMarkerType to
    //!            avoid loosing entries due to internal
    //!            resizing.
    //! \note Depending on the marker type the colors
    //!       are assigned indexwise to the points or
    //!       to the elements as lines or line strip.
    //! \note If too many colors are provided, the
    //!       color vector will be resized. If \a newColor
    //!       is empty, the color vector will set to
    //!       hold one element of the default color, to
    //!       the vector will always contain at least one
    //!       color.
    //! \attention Using setColor will overwrite the
    //!            first color entry.
    //----------------------------------------
    uint32_t setColorVector(ColorVector&& newColors)
    {
        m_colors = std::move(newColors);
        normalize(m_markerType, m_points, m_colors);
        return static_cast<uint32_t>(m_points.size());
    }

    //========================================
    //! \brief Set a new marker's text.
    //! \param[in] newText  The new marker's text.
    //----------------------------------------
    void setText(const std::string& newText) { m_text = newText; }

    //========================================
    //! \brief Set a new marker's text.
    //! \param[in, out] newText  The new marker's text.
    //----------------------------------------
    void setText(std::string&& newText) { m_text = std::move(newText); }

    //========================================
    //! \brief Set the new marker's lifetime in milliseconds.
    //! \param[in] newLifetimeInMilliseconds  New marker's lifetime in milliseconds.
    //----------------------------------------
    void setLifetimeInMilliseconds(const uint32_t newLifetimeInMilliseconds)
    {
        m_lifetimeInMilliseconds = newLifetimeInMilliseconds;
    }

    //========================================
    //! \brief Set whether the marker is locked to the
    //!        coordinate system while its lifetime.
    //! \param[in] locked  Whether the marker is locked to
    //!                    the coordinate system while its
    //!                    lifetime.
    //----------------------------------------
    void setIsCoordinateSystemLockedWhileLifetime(const bool locked)
    {
        m_isCoordinateSystemLockedWhileLifetime = locked;
    }

    //========================================
    //! \brief Set if the marker namespace is
    //!        complete.
    //!
    //! Set if the marker namespace is complete
    //! without the marker list's namespace prefix.
    //!
    //! \param[in] isComplete  Whether the namespace is
    //!                        complete without the marker
    //!                        list namespace prefix.
    //----------------------------------------

    void setMarkerNamespaceComplete(const bool isComplete) { m_isNamespaceComplete = isComplete; }

    //========================================
    //! \brief Set the marker's namespace.
    //! \param[in] newMarkerNamespace  The new marker's namespace.
    //----------------------------------------
    void setMarkerNamespace(const MarkerNamespace& newMarkerNamespace) { m_markerNamespace = newMarkerNamespace; }

    //========================================
    //! \brief Set the marker's namespace.
    //! \param[in,out] newMarkerNamespace  The new marker's namespace.
    //----------------------------------------
    void setMarkerNamespace(MarkerNamespace&& newMarkerNamespace) { m_markerNamespace = std::move(newMarkerNamespace); }

    //========================================
    //! \brief Set the marker's ID.
    //! \param[in] newMarkerId  The new marker's ID.
    //----------------------------------------
    void setMarkerId(const int32_t newMarkerId) { m_markerId = newMarkerId; }

protected:
    //========================================
    //! \brief Depending on the marker type it contains
    //!        vertices or not.
    //! \return \c True if the marker type is using
    //!         the point array. \c false otherwise.
    //! \note MarkerType::Arrow is a special case, it
    //!       may or may not make use of the point
    //!       vector. Since it decides its behavior on
    //!       the size of the points vector, \c true
    //!       will be returned for MarkerType::Arrow.
    //!       If the point vector is not used, the
    //!       serialization of it will be completely skipped.
    //! \note Also the serialization of m_colors is
    //!       affected by this value. If \c false, the
    //!       first (and only) m_colors entry will be
    //!       not serialized as vector of colors but
    //!       just as a single color.
    //----------------------------------------
    bool markerTypeToWhetherUsingVertices() const;

protected:
    //========================================
    //! \brief Depending on the marker type, unused
    //!        vertices and colors will be removed
    //!        from the point and color vector.
    //! \param[in] type             The type of the marker to be normalized.
    //! \param[in,out] pointVector  The marker's point vector to be modified,
    //!                             if necessary.
    //! \param[in,out] colorVector  The marker's color vector to be modified,
    //!                             if necessary.
    //----------------------------------------
    static void normalize(const MarkerType type, PointVector& pointVector, ColorVector& colorVector);

    //========================================
    //! \brief Cleanup the points vector and colors
    //!        vector for marker types which are not
    //!        using the points array.
    //!
    //! Clear the points vector and all colors but one.
    //! \param[in,out] pointVector  The point vector to be cleared,
    //!                             if it is not empty.
    //! \param[in,out] colorVector  The color vector to be resized to size 1,
    //!                             if necessary.
    //----------------------------------------
    static void normalizeNoVertices(PointVector& pointVector, ColorVector& colorVector);

    //========================================
    //! \brief Ensure the number of points and colors fitting
    //!        for the Arrow marker type.
    //! \param[in,out] pointVector    The point vector to be checked and to be
    //!                               resized, if necessary.
    //! \param[in,out] colorVector    The color vector to be resized to size 1,
    //!                               if necessary.
    //! An arrow can have 0 or 2 points and has 1 color.
    //----------------------------------------
    static void normalizeArrow(PointVector& pointVector, ColorVector& colorVector);

    //========================================
    //! \brief On exit the \a pointVector has at least \a minNbOfPoints or is empty.
    //! \param[in,out] pointVector    The point vector to be checked and to be
    //!                               resized if necessary.
    //! \param[in] minNbOfPoints      The minimal number of points for a valid
    //!                               non-empty point vector.
    //----------------------------------------
    static void ensureToBeAtLeastOrEmpty(PointVector& pointVector, const uint32_t minNbOfPoints);

    //========================================
    //! \brief Ensure that \a factor is a divider of the size of pointVector.
    //! \param[in,out] pointVector  The point vector to be checked and to be
    //!                             resized if necessary.
    //!                             On exit the number of points is a multiple of
    //!                             \a factor.
    //! \param[in] factor           On exit the number of points is a multiple of
    //!                             \a factor.
    //----------------------------------------
    static void shrinkToBeMultipleOf(PointVector& pointVector, const uint32_t factor);

    //========================================
    //! \brief On exit the \a colorVector is not bigger than the maximum of \a maxNbOfColors,
    //!        and 1.
    //! \param[in] colorVector   Color vector to be resized if necessary.
    //! \param[in] maxNbOfColor  If maxNbOfColor is greater or equal 1, \a colorVector will be
    //!                          resized to \a maxNbOfColor elements, if necessary.
    //----------------------------------------
    static void ensureToBeNotMoreThanButNotEmpty(ColorVector& colorVector, const uint32_t maxNbOfColors);

protected:
    //========================================
    //! \brief Deserialize a string \a targetString from \a is.
    //!
    //! The length of the string is serialized before the string content as
    //! a uint16_t.
    //!
    //! \param[in,out] is            Input stream, the length of the following string
    //!                              and the string itself is read from.
    //! \param[out]    targetString  The string, where the result of the
    //!                              deserialzation is stored.
    //----------------------------------------
    static void deserializeString(std::istream& is, std::string& targetString);

    //========================================
    //! \brief Serialize a string \a targetString to \a os.
    //!
    //! The length of the string is serialized before the string content as
    //! a uint16_t.

    //! \param[in,out] os            Output stream, the length of the string and the
    //!                              string is serialized to.
    //! \param[in]     sourceString  The string to be serialized.
    //----------------------------------------
    static void serializeString(std::ostream& os, const std::string& sourceString);

protected:
    //========================================
    //! \brief The marker's type.
    //----------------------------------------
    MarkerType m_markerType{MarkerType::Points};

    //========================================
    //! \brief Internal variable, holds the (deserialized)
    //!        information, if vertices are used or not.
    //----------------------------------------
    bool m_markerIsUsingVertices{false};

    //========================================
    //! \brief The information, whether the marker is locked
    //!        to the coordinate system while its lifetime.
    //----------------------------------------
    bool m_isCoordinateSystemLockedWhileLifetime{false};

    //========================================
    //! \brief The information, whether the namespace
    //!        is complete.
    //!
    //! This flag holds the information, whether the
    //! namespace is complete without the namespace
    //! prefix in the marker list.
    //----------------------------------------
    bool m_isNamespaceComplete{true};

    //========================================
    //! \brief The marker's namespace.
    //!
    //! The namespace is represented as a vector
    //! of strings to be read:
    //!
    //! String[0]::String[1]::String[2]...
    //----------------------------------------
    MarkerNamespace m_markerNamespace{};

    //========================================
    //! \brief The marker's position (or offset).
    //----------------------------------------
    Position m_position{};

    //========================================
    //! \brief The orientation of the marker's coordinate system.
    //----------------------------------------
    Orientation m_orientation{};

    //========================================
    //! \brief Scaling factors used by some marker types.
    //----------------------------------------
    Scale m_scale{};

    //========================================
    //! \brief A vector of vertices/points used by some marker types.
    //----------------------------------------
    PointVector m_points{};

    //========================================
    //! \brief A vector of colors used by the marker.
    //!
    //! This vector is guaranteed to be not empty.
    //----------------------------------------
    ColorVector m_colors; // cannot be declared here due to link problems with Visual Studio.

    //========================================
    //! \brief The text of this marker.
    //----------------------------------------
    std::string m_text{};

    //========================================
    //! \brief The ID of this marker.
    //!
    //! It is expected that the combination of
    //! markerId and namespace is unique.
    //----------------------------------------
    int32_t m_markerId{};

    //========================================
    //! \brief The lifetime of this marker in milliseconds.
    //----------------------------------------
    uint32_t m_lifetimeInMilliseconds{40};

    //========================================
    //! \brief An array of reserved attributes.
    //!
    //! This vector is empty unless this marker
    //! has been deserialized from a future version
    //! of the marker deserialization.
    //----------------------------------------
    std::vector<char> m_reserved{};
}; // MarkerIn6820

//==============================================================================

template<>
inline void readLE<MarkerIn6820::MarkerType>(std::istream& is, MarkerIn6820::MarkerType& value)
{
    uint8_t tmp;
    readBE(is, tmp);
    value = static_cast<MarkerIn6820::MarkerType>(tmp);
}

//==============================================================================

template<>
inline void writeLE<MarkerIn6820::MarkerType>(std::ostream& os, const MarkerIn6820::MarkerType& value)
{
    const uint8_t tmp = static_cast<uint8_t>(value);
    writeBE(os, tmp);
}

//==============================================================================
// operators
//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const MarkerIn6820& lhs, const MarkerIn6820& rhs);

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const MarkerIn6820& lhs, const MarkerIn6820& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
