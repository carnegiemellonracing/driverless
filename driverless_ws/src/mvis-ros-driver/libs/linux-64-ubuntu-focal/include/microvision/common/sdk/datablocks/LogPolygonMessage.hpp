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
//! \date May 03, 2019
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <string>
#include <unordered_map>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//!\brief The message part used in a LogPolygon.
//!
//! LogPolygonMessages have the following format
//!
//! \c [TEXT]###[[[Property=value],Property=value],...]
//!
//! Where TEXT is the text of the message which cannot contain a ###.
//! After the ### separator there can be a list of comma separated
//! property-values pairs. In the property section ALL spaces will be ignored.
//! TEXT and the properties can be an empty string. If the ### separator is missing
//! the LogPolygonMessage contains only the TEXT part.
//------------------------------------------------------------------------------
class LogPolygonMessage final
{
public:
    using LogMsgProperties        = std::unordered_map<std::string, std::string>;
    using PropertyIndexMap        = std::unordered_map<uint64_t, size_t>;
    using PropertySubScriptVector = std::vector<std::pair<std::size_t, std::size_t>>;

public:
    static void parseRawMsg(std::string& logMsgText, std::size_t& startOfProperties, const std::string& rawMsg);

    static void parseProperties(std::string& rawPropertyString,
                                PropertyIndexMap& propIndexMap,
                                PropertySubScriptVector& propValueStringBoundariesVector,
                                const std::string& rawMsg,
                                const std::size_t startOfProperties);

    static std::size_t addNextProperty(PropertyIndexMap& propIndexMap,
                                       PropertySubScriptVector& propValueStringBoundariesVector,
                                       const std::string& rawPropertyString,
                                       const std::size_t start,
                                       const std::size_t idx);

public:
    LogPolygonMessage() = default;
    LogPolygonMessage(const std::string& rawMsg);
    ~LogPolygonMessage() = default;

public:
    void setRawMessage(const std::string& rawMsg)
    {
        m_rawMsg             = rawMsg;
        m_isParsed           = false;
        m_arePropertiesParse = false;
    }
    void setRawMessage(std::string&& rawMsg)
    {
        m_rawMsg             = std::move(rawMsg);
        m_isParsed           = false;
        m_arePropertiesParse = false;
    }

    const std::string& getRawMessage() const { return m_rawMsg; }

    std::string getMsgText() const
    {
        if (!m_isParsed)
        {
            parseRawMsg();
        }
        return m_logMsgText;
    }

    bool getMsgProperty(std::string& propertyValue, const uint64_t propertyHash) const;

private:
    //========================================
    //!\brief Parse the m_rawMsg for its text part and properties.
    //!
    //! The text part will be stored in m_logMsgText.
    //! The properties will be sorted in an unordered map m_logMsgProperties.
    //!
    //! \note If a property is given twice, the second occurrence will overwrite the
    //!       first.
    //! \note The Text-Property separator "###" is expected to be in the raw string
    //!       only once, however any further occurrence will be replaced by a comma.
    //! \note For a property only the first equal sign will separate property key and
    //!       property value. Further equal signs immediately following the first will
    //!       be ignored, others will become part of the property value.
    //! \note By MicroVision reserved property keys are "alpha", "color", "filled", "group",
    //!       "pointSize", "width", "zIndex", "zOffset".
    //----------------------------------------
    void parseRawMsg() const;
    void parseProperties() const;

private:
    std::string m_rawMsg;

    mutable bool m_isParsed{false};
    mutable bool m_arePropertiesParse{false};
    mutable std::size_t m_startOfProperties{0};

    mutable std::string m_logMsgText;
    mutable std::string m_rawPropertyString;
    mutable PropertyIndexMap m_propIndexMap;
    mutable PropertySubScriptVector m_propValueStringBoundariesVector;

}; // LogPolygonMessage

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
