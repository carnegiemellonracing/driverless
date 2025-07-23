//==============================================================================
//! \file
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <cstdint>
#include <string>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace usereventtag7010types {
//==============================================================================

//==============================================================================
//!\brief The type of event.
//------------------------------------------------------------------------------
enum class EventCategory : uint8_t
{
    Comment       = 0x00U, //!< This event solely contains a message.
    UserDefined   = 0x01U, //!< This tag was defined by the user.
    LaneCount     = 0x10U, //!< Number of lanes.
    RoadType      = 0x11U, //!< Layout of the road.
    RoadCondition = 0x12U, //!< Characteristics of the road.
    Weather       = 0x30U, //!< Environmental conditions.
    Action        = 0x40U, //!< Behavior of road users.
    Undefined     = 0xFFU, //!< The category is undefined.
}; // EventCategory

//==============================================================================
//!\brief Qualifies the appearance of the event.
//------------------------------------------------------------------------------
enum class TagOccurrence : uint8_t
{
    Start     = 0x10U, //!< Beginning of a situation.
    End       = 0x11U, //!< End of a situation.
    Pulse     = 0x20U, //!< Single point in time.
    Undefined = 0xFFU, //!< Undefined.
};

//==============================================================================
//!\brief Qualifies the intensity of the event
//------------------------------------------------------------------------------
enum class TagSeverity : uint8_t
{
    Normal    = 0x00U, //!< The intensity is normal.
    Light     = 0x10U, //!< The intensity is light.
    Medium    = 0x11U, //!< The intensity is medium.
    Heavy     = 0x12U, //!< The intensity is heavy.
    Undefined = 0xFFU, //!< The intensity is undefined.
};

//==============================================================================
//!\brief The type of road.
//------------------------------------------------------------------------------
enum class EventRoadType : uint8_t
{
    Motorway     = 0x10U, //!< Motorway / Autobahn
    CountryRoad  = 0x11U, //!< Outer city country road
    StateRoad    = 0x12U, //!< Outer city greater road
    InnerCity    = 0x13U, //!< Inner city street
    Exit         = 0x20U, //!< Motorway or property exit
    Driveway     = 0x21U, //!< Motorway or property entrance
    Intersection = 0x22U, //!< Crossing
    Tunnel       = 0x30U, //!< Under bridge
    Undefined    = 0xFFU, //!< Undefined
};

//
//==============================================================================
//!\brief The condition of the road.
//------------------------------------------------------------------------------
enum class EventRoadCondition : uint8_t
{
    RoadWorks = 0x10U, //!< Construction side
    Dirt      = 0x11U, //!< Polluted road
    Undefined = 0xFFU, //!< Undefined
};

//
//==============================================================================
//!\brief The environmental situation (weather).
//------------------------------------------------------------------------------
enum class EventWeather : uint8_t
{
    Rain      = 0x10U, //!< It is raining.
    Snow      = 0x11U, //!< It is snowing.
    Fog       = 0x12U, //!< There is fog.
    Undefined = 0xFFU, //!< Undefined.
};

//==============================================================================
//!\brief Another car's movement relative to us.
//------------------------------------------------------------------------------
enum class EventAction : uint8_t
{
    CutIn      = 0x10U, //!< Another car enters our lane
    CutOut     = 0x11U, //!< Another car leaves our lane
    Congestion = 0x20U, //!< Traffic jam
    Undefined  = 0xFFU //!< Undefined.
};

//==============================================================================

template<typename T>
inline std::string toString(const T) // enumeration
{
    return "Unknown UserEventTag7010Type";
}

//==============================================================================

// We only allow these explicit template specializations for our tag and event enumerations.
template<>
std::string inline toString<EventCategory>(const EventCategory category)
{
    switch (category)
    {
    case EventCategory::Comment:
        return "Comment";
    case EventCategory::UserDefined:
        return "User defined";
    case EventCategory::LaneCount:
        return "Lane Count";
    case EventCategory::RoadType:
        return "Road Type";
    case EventCategory::RoadCondition:
        return "Road Condition";
    case EventCategory::Weather:
        return "Weather";
    case EventCategory::Action:
        return "Action";
    case EventCategory::Undefined:
        // fall-thru
    default:
        return "Undefined Category";
    }
}

//==============================================================================

template<>
std::string inline toString<EventRoadType>(const EventRoadType type)
{
    switch (type)
    {
    case EventRoadType::Motorway:
        return "Motorway";
    case EventRoadType::CountryRoad:
        return "Country Road";
    case EventRoadType::StateRoad:
        return "State Road";
    case EventRoadType::InnerCity:
        return "Inner City";
    case EventRoadType::Exit:
        return "Exit";
    case EventRoadType::Driveway:
        return "Entrance";
    case EventRoadType::Intersection:
        return "Intersection";
    case EventRoadType::Tunnel:
        return "Tunnel";
    case EventRoadType::Undefined:
        // fall-thru
    default:
        return "Undefined Road Type";
    }
}

//==============================================================================

template<>
std::string inline toString<EventRoadCondition>(const EventRoadCondition condition)
{
    switch (condition)
    {
    case EventRoadCondition::RoadWorks:
        return "Road Works";
    case EventRoadCondition::Dirt:
        return "Dirty Road";
    case EventRoadCondition::Undefined:
        // fall-thru
    default:
        return "Undefined Road Condition";
    }
}

//==============================================================================

template<>
std::string inline toString<EventWeather>(const EventWeather weather)
{
    switch (weather)
    {
    case EventWeather::Rain:
        return "Rain";
    case EventWeather::Snow:
        return "Snow";
    case EventWeather::Fog:
        return "Fog";
    case EventWeather::Undefined:
        // fall-thru
    default:
        return "Undefined Weather Condition";
    }
}

//==============================================================================

template<>
std::string inline toString<EventAction>(const EventAction enumeration)
{
    switch (enumeration)
    {
    case EventAction::CutIn:
        return "Cut In";
    case EventAction::CutOut:
        return "Cut Out";
    case EventAction::Congestion:
        return "Traffic Jam";
    case EventAction::Undefined:
        // fall-thru
    default:
        return "Undefined Action";
    }
}

//==============================================================================

template<>
std::string inline toString<TagOccurrence>(const TagOccurrence occurrence)
{
    switch (occurrence)
    {
    case TagOccurrence::Start:
        return "Start of Event";
    case TagOccurrence::End:
        return "End of Event";
    case TagOccurrence::Pulse:
        return "Pulse Event";
    case TagOccurrence::Undefined:
        // fall-thru
    default:
        return "Undefined Occurrence";
    }
}

//==============================================================================

template<>
std::string inline toString<TagSeverity>(const TagSeverity severity)
{
    switch (severity)
    {
    case TagSeverity::Normal:
        return "Normal Severity";
    case TagSeverity::Light:
        return "Light Severity";
    case TagSeverity::Medium:
        return "Medium Severity";
    case TagSeverity::Heavy:
        return "Heavy Severity";
    case TagSeverity::Undefined:
        // fall-thru
    default:
        return "Undefined Severity";
    }
}

//==============================================================================

template<>
std::string inline toString<uint8_t>(const uint8_t value)
{
    // uint8_t == uchar ...
    return std::to_string(static_cast<uint64_t>(value));
}

//==============================================================================
} // namespace usereventtag7010types
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
