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
//! \date Oct 9, 2014
//! \brief Definition of structs and enums for carriageways and lanes
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================
//!\brief Enumeration with different CarriageWay types
//------------------------------------------------------------------------------
enum class CarriageWayType : uint8_t
{
    Motorway = 0, //!< Used for highways.

    //! Used for highway similar roads (Germany: Autobahnaehnliche Strasse, umgangssprachlich: Schnellstrasse).
    Trunk        = 1,
    Primary      = 2, //!< Used for primary roads (Germany: Bundesstrasse).
    Secondary    = 3, //!< Used for secondary roads (Germany: Landesstrasse).
    Tertiary     = 4, //!< Used for tertiary roads (Germany: Kreisstrasse).
    Residential  = 5, //!< Used for residential roads (Germany: Wohngebiet).
    Service      = 6, //!< Used for service roads (Germany: Zufahrtsstrasse, Seitenstrasse).
    Unclassified = 7 //!< Used if road is not classifiable.
};

//==============================================================================
//!\brief Enumeration for Lane types
//------------------------------------------------------------------------------
enum class LaneType : uint8_t
{
    Forward      = 0, //!< Driving direction is forward
    Backward     = 1, //!< Driving direction id backward
    Exit         = 2, //!< Exit Lane (Germany: Ausfahrt)
    Oncoming     = 3, //!< Oncoming Lane (German: Beschleunigungsspur)
    Breakdown    = 4, //!< Breakdown Lane (Germany: Standspur)
    Unclassified = 5, //!< Not classifiable
    Invalid      = 6 //!< Invalid lane to incorporate curbstones/guardrails into carriageway
};

//==============================================================================
//!\brief Enumeration for Lane Markings
//------------------------------------------------------------------------------

enum class LaneMarkingType : uint8_t
{
    Unclassified = (1 << 1), //!< Not classifiable
    Single       = (1 << 2), //!< Single marking
    Double       = (1 << 3), //!< Double marking
    Solid        = (1 << 4), //!< Solid marking
    Dashed       = (1 << 5), //!< Dashed marking
    None         = (1 << 6) //!< No marking at all
};

//==============================================================================
// !-\brief Enumeration for Border types
//------------------------------------------------------------------------------
enum class BorderType : uint8_t
{
    LaneChangePossible   = 0, //!< A neighboring lane exist and a Lane change is possible
    NoLaneChangePossible = 1, //!< A neighboring lane exist but no Lane change is possible (e.g. due to guard rails)
    OncomingLane         = 2, //!< An on coming Lane
    EndOfStreetSolid     = 3, //!< End of the road of solid nature (e.g. guard rail, buildings)
    EndOfStreetEarth     = 4, //!< End of the road of planar and accessible nature (e.g. earth, asphalt)
    Unclassified         = 5 //!< Unclassified border
};

//==============================================================================
//!\brief Enumeration for Lane Boundaries
//------------------------------------------------------------------------------
enum class LaneBoundaryType : uint8_t
{
    Lane_Marking = 0, //!< Simple (painted) lane marking
    Offroad      = 1, //!< Off-road, no lane boundary available at all
    Curbstone    = 2, //!< Curbstone as boundary
    Guardrails   = 3, //!< Guardrails
    Unclassified = 5 //!< Boundary type not classifiable
};

//==============================================================================
//!\brief Structure for holding a bounding rectangle in geographic coordinates
//------------------------------------------------------------------------------
struct BoundingRectangle
{
public:
    //! \brief constructor which initializes with virtual coordinates which are
    //! out of range (+-1000) so the will be overwritten when checking real boundaries
    // ------------------------------------------------------------------------------
    BoundingRectangle() : minLongitude(1000), maxLongitude(-1000), minLatitude(1000), maxLatitude(-1000) {}

public:
    bool checkInside(const PositionWgs84& point) const
    {
        return (point.getLatitudeInDeg() <= this->maxLatitude && point.getLatitudeInDeg() >= this->minLatitude
                && point.getLongitudeInDeg() >= this->minLongitude && point.getLongitudeInDeg() <= this->maxLongitude);
    }

    void expand(const BoundingRectangle& other)
    {
        minLatitude  = (other.minLatitude >= minLatitude) ? minLatitude : other.minLatitude;
        maxLatitude  = (other.maxLatitude <= maxLatitude) ? maxLatitude : other.maxLatitude;
        minLongitude = (other.minLongitude >= minLongitude) ? minLongitude : other.minLongitude;
        maxLongitude = (other.maxLongitude <= maxLongitude) ? maxLongitude : other.maxLongitude;
    }

public:
    double minLongitude; //!< Minimal longitude position in degrees
    double maxLongitude; //!< Maximal longitude position in degrees
    double minLatitude; //!< Minimal latitude position in degrees
    double maxLatitude; //!< Maximal latitude position in degrees
}; // BoundingRectangle

//==============================================================================

inline bool operator==(const BoundingRectangle& lhs, const BoundingRectangle& rhs)
{
    return fuzzyDoubleEqualT<7>(lhs.minLongitude, rhs.minLongitude) //
           && fuzzyDoubleEqualT<7>(lhs.maxLongitude, rhs.maxLongitude) //
           && fuzzyDoubleEqualT<7>(lhs.minLatitude, rhs.minLatitude) //
           && fuzzyDoubleEqualT<7>(lhs.maxLatitude, rhs.maxLatitude);
}

//==============================================================================

inline bool operator!=(const BoundingRectangle& lhs, const BoundingRectangle& rhs) { return !(lhs == rhs); }

//==============================================================================
//!\brief Constructor for std::map initialization, since const private
//!       maps need to be initialized by Constructor or function within
//!       C++98 standard.
//------------------------------------------------------------------------------
struct MapConstructor
{
public:
    //========================================
    //!\brief Creates the map from std::string to \ref CarriageWayType.
    //----------------------------------------
    static std::map<std::string, CarriageWayType> createCWTMap()
    {
        std::map<std::string, CarriageWayType> map;
        map["motorway"]     = CarriageWayType::Motorway;
        map["trunk"]        = CarriageWayType::Trunk;
        map["primary"]      = CarriageWayType::Primary;
        map["secondary"]    = CarriageWayType::Secondary;
        map["tertiary"]     = CarriageWayType::Tertiary;
        map["residential"]  = CarriageWayType::Residential;
        map["service"]      = CarriageWayType::Service;
        map["unclassified"] = CarriageWayType::Unclassified;
        return map;
    }

    //========================================
    //!\brief Creates the map from std::string to \ref LaneType.
    //----------------------------------------
    static std::map<std::string, LaneType> createLTMap()
    {
        std::map<std::string, LaneType> map;
        map["forward"]      = LaneType::Forward;
        map["backward"]     = LaneType::Backward;
        map["exit"]         = LaneType::Exit;
        map["oncoming"]     = LaneType::Oncoming;
        map["breakdown"]    = LaneType::Breakdown;
        map["unclassified"] = LaneType::Unclassified;
        return map;
    }

    //========================================
    //!\brief Creates the map from std::string to \ref LaneMarkingType.
    //----------------------------------------
    static std::map<std::string, LaneMarkingType> createLMTMap()
    {
        std::map<std::string, LaneMarkingType> map;
        map["unclassified"] = LaneMarkingType::Unclassified;
        map["single"]       = LaneMarkingType::Single;
        map["double"]       = LaneMarkingType::Double;
        map["solid"]        = LaneMarkingType::Solid;
        map["dashed"]       = LaneMarkingType::Dashed;
        map["none"]         = LaneMarkingType::None;
        return map;
    }

    //========================================
    //!\brief Creates a map from std::string to \ref BorderType
    //----------------------------------------
    static std::map<std::string, BorderType> createBorderMap()
    {
        std::map<std::string, BorderType> map;
        map["unclassified"]         = BorderType::Unclassified;
        map["lanechangepossible"]   = BorderType::LaneChangePossible;
        map["lanenochangepossible"] = BorderType::NoLaneChangePossible;
        map["oncominglane"]         = BorderType::OncomingLane;
        map["endofstreetsolid"]     = BorderType::EndOfStreetSolid;
        map["endofstreetearth"]     = BorderType::EndOfStreetEarth;
        return map;
    }

    //========================================
    //!\brief Creates a map from std::string to \ref createBoundaryMap
    //----------------------------------------
    static std::map<std::string, LaneBoundaryType> createBoundaryMap()
    {
        std::map<std::string, LaneBoundaryType> map;
        map["lanemarking"]  = LaneBoundaryType::Lane_Marking;
        map["offroad"]      = LaneBoundaryType::Offroad;
        map["curbstone"]    = LaneBoundaryType::Curbstone;
        map["guardrails"]   = LaneBoundaryType::Guardrails;
        map["unclassified"] = LaneBoundaryType::Unclassified;

        return map;
    }

    static std::map<std::string, bool> createBoolMap()
    {
        std::map<std::string, bool> map;
        map["true"]  = true;
        map["false"] = false;
        return map;
    }

    static std::map<CarriageWayType, std::string> createFromCWTMap()
    {
        std::map<CarriageWayType, std::string> map;
        map[CarriageWayType::Motorway]     = "motorway";
        map[CarriageWayType::Trunk]        = "trunk";
        map[CarriageWayType::Primary]      = "primary";
        map[CarriageWayType::Secondary]    = "secondary";
        map[CarriageWayType::Tertiary]     = "tertiary";
        map[CarriageWayType::Residential]  = "residential";
        map[CarriageWayType::Service]      = "service";
        map[CarriageWayType::Unclassified] = "unclassified";
        return map;
    }

    static std::map<LaneType, std::string> createLTToStringMap()
    {
        std::map<LaneType, std::string> map;
        map[LaneType::Forward]      = "forward";
        map[LaneType::Backward]     = "backward";
        map[LaneType::Exit]         = "exit";
        map[LaneType::Oncoming]     = "oncoming";
        map[LaneType::Breakdown]    = "breakdown";
        map[LaneType::Unclassified] = "unclassified";
        return map;
    }

    static std::map<LaneMarkingType, std::string> createLMTToStringMap()
    {
        std::map<LaneMarkingType, std::string> map;
        map[LaneMarkingType::Unclassified] = "unclassified";
        map[LaneMarkingType::Single]       = "single";
        map[LaneMarkingType::Double]       = "double";
        map[LaneMarkingType::Solid]        = "solid";
        map[LaneMarkingType::Dashed]       = "dashed";
        map[LaneMarkingType::None]         = "none";
        return map;
    }

    //	static std::map<BorderType, std::string> createBorderToStringMap()
    //	{
    //		std::map<BorderType, std::string> map;
    //		map[BorderType::Unclassified]         = "unclassified";
    //		map[BorderType::LaneChangePossible]   = "lanechangepossible";
    //		map[BorderType::NoLaneChangePossible] = "lanenochangepossible";
    //		map[BorderType::OncomingLane]         = "oncominglane";
    //		map[BorderType::EndOfStreetSolid]     = "endofstreetsolid";
    //		map[BorderType::EndOfStreetEarth]     = "endofstreetearth";
    //		return map;
    //	}

    static std::map<LaneBoundaryType, std::string> createBoundaryToStringMap()
    {
        std::map<LaneBoundaryType, std::string> map;
        map[LaneBoundaryType::Lane_Marking] = "lanemarking";
        map[LaneBoundaryType::Offroad]      = "offroad";
        map[LaneBoundaryType::Curbstone]    = "curbstone";
        map[LaneBoundaryType::Guardrails]   = "guardrails";
        map[LaneBoundaryType::Unclassified] = "unclassified";
        return map;
    }

    static std::map<bool, std::string> createFromBoolMap()
    {
        std::map<bool, std::string> map;
        map[true]  = "true";
        map[false] = "false";
        return map;
    }
}; // MapConstructor

//==============================================================================

using mapConstructor = MapConstructor;

//==============================================================================

const std::map<std::string, CarriageWayType> CarriageWayTypeFromStringMap = MapConstructor::createCWTMap();

const std::map<CarriageWayType, std::string> CarriageWayTypeToStringMap = MapConstructor::createFromCWTMap();

const std::map<std::string, LaneType> LaneTypeFromStringMap = MapConstructor::createLTMap();

const std::map<LaneType, std::string> LaneTypeToStringMap = MapConstructor::createLTToStringMap();

const std::map<std::string, LaneMarkingType> LaneMarkingTypeFromStringMap = MapConstructor::createLMTMap();

const std::map<LaneMarkingType, std::string> LaneMarkingTypeToStringMap = MapConstructor::createLMTToStringMap();

const std::map<std::string, LaneBoundaryType> BorderTypeFromStringMap = MapConstructor::createBoundaryMap();

const std::map<LaneBoundaryType, std::string> BorderTypeToStringMap = MapConstructor::createBoundaryToStringMap();

const std::map<std::string, bool> BoolFromStringMap = MapConstructor::createBoolMap();

const std::map<bool, std::string> BoolToStringMap = MapConstructor::createFromBoolMap();

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
