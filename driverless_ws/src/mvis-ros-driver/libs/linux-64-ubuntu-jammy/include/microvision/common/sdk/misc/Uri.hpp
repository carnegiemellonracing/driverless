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
//! \date Jun 5, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <string>
#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Uri supported transfer protocols.
//!
//! Uri supported transfer protocols.
//------------------------------------------------------------------------------
enum class UriProtocol : uint8_t
{
    NoAddr, //!< Not addressable resource.
    Tcp, //!< Eth tcp transfer protocol.
    Udp, //!< Eth udp transfer protocol.
    File, //!< Local/Network accessible file.
};

//==============================================================================
//! \brief Parameters to define a Uri.
//!
//! Parameters to define a Uri.
//------------------------------------------------------------------------------
class Uri final
{
public:
    //========================================
    //! \brief URL separator after protocol.
    //----------------------------------------
    static MICROVISION_SDK_API constexpr const char* protocolSeparator{"://"};

    //========================================
    //! \brief URL Separator between host and port.
    //----------------------------------------
    static MICROVISION_SDK_API constexpr char portSeparator{':'};

    //========================================
    //! \brief URL Separator of path.
    //----------------------------------------
    static MICROVISION_SDK_API constexpr char pathSeparator{'/'};

    //========================================
    //! \brief Get URL Separator after protocol.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getProtocolSeparator()
    {
        static const std::string Separator{protocolSeparator};
        return Separator;
    }

    //========================================
    //! \brief Get URL Separator between host and port.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getPortSeparator()
    {
        static const std::string Separator{portSeparator};
        return Separator;
    }

    //========================================
    //! \brief Get URL separator of path.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getPathSeparator()
    {
        static const std::string Separator{pathSeparator};
        return Separator;
    }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    Uri() = default;

    //========================================
    //! \brief Constructs file uri.
    //! \param[in] filePath  Path to file on localhost.
    //----------------------------------------
    Uri(const std::string& filePath) : m_proto{UriProtocol::File}, m_path{filePath} {}

    //========================================
    //! \brief Constructs network uri.
    //! \param[in] proto    UriProtocol, expects \c Tcp or \c Udp.
    //! \param[in] host     Host identifier like ipv4 address.
    //! \param[in] port     Host port number.
    //----------------------------------------
    Uri(const UriProtocol proto, const std::string& host, const uint16_t port)
      : m_proto{proto}, m_host{host}, m_port{port}
    {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Uri() = default;

public: //getter
    //========================================
    //! \brief Get's a format id like MDF/IDC etc.
    //! \return Get a value copy of the format id.
    //----------------------------------------
    std::string getFormat() const noexcept { return m_format; }

    //========================================
    //! \brief Get's a version of the format like 4.1.11 etc.
    //! \return Get a value copy of the version.
    //----------------------------------------
    std::string getVersion() const noexcept { return m_version; }

    //========================================
    //! \brief Get's a format schema like idc/pcap (in case of MDF) etc.
    //! \return Get a value copy of the schema.
    //----------------------------------------
    std::string getSchema() const noexcept { return m_schema; }

    //========================================
    //! \brief Get's a transfer protocol.
    //! \return Get a value copy of the transfer protocol.
    //----------------------------------------
    UriProtocol getProtocol() const noexcept { return m_proto; }

    //========================================
    //! \brief Get's a host by name or ip.
    //! \return Get a value copy of the name/ip.
    //----------------------------------------
    std::string getHost() const noexcept { return m_host; }

    //========================================
    //! \brief Get's the host port like 80.
    //! \return Get a value copy of the port.
    //----------------------------------------
    uint16_t getPort() const noexcept { return m_port; }

    //========================================
    //! \brief Get's the file/url path for the host.
    //! \return Get a value copy of the path.
    //----------------------------------------
    std::string getPath() const noexcept { return m_path; }

public: //setter
    //========================================
    //! \brief Set the format like MDF/IDC etc.
    //! \param[in] format  Format like MDF/IDC.
    //----------------------------------------
    void setFormat(const std::string& format) { m_format = format; }

    //========================================
    //! \brief Set the format version like 4.1.11 etc.
    //! \param[in] version  Format version like 1.1.1.
    //----------------------------------------
    void setVersion(const std::string& version) { m_version = version; }

    //========================================
    //! \brief Set the format schema like idc/pcap (in case of MDF) etc.
    //! \param[in] schema  Format schema like IDC for format MDF.
    //----------------------------------------
    void setSchema(const std::string& schema) { m_schema = schema; }

    //========================================
    //! \brief Set the uri protocol.
    //! \param[in] proto  Uri protocol.
    //----------------------------------------
    void setProtocol(const UriProtocol& proto) { m_proto = proto; }

    //========================================
    //! \brief Set the host name/ip.
    //! \param[in] host  Host name or ip.
    //----------------------------------------
    void setHost(const std::string& host) { m_host = host; }

    //========================================
    //! \brief Set the port like 80.
    //! \param[in] port  Port of host.
    //----------------------------------------
    void setPort(const uint16_t& port) { m_port = port; }

    //========================================
    //! \brief Set the file/url path for the host.
    //! \param[in] path  File/URL path for host.
    //----------------------------------------
    void setPath(const std::string& path) { m_path = path; }

public: // operators
    //========================================
    //! \brief Print Uri as string into output stream.
    //! \param[in] stream   Output stream to write.
    //! \param[in] uri      Uri to print.
    //! \return The input parameter stream will get back for operator chain.
    //----------------------------------------
    friend std::ostream& operator<<(std::ostream& stream, const Uri& uri)
    {
        if (!uri.getFormat().empty())
        {
            stream << uri.getFormat();

            if (!uri.getSchema().empty())
            {
                stream << "." << uri.getSchema();
            }
            if (!uri.getVersion().empty())
            {
                stream << "(" << uri.getVersion() << ")";
            }

            stream << " ";
        }

        switch (uri.getProtocol())
        {
        case UriProtocol::NoAddr:
            stream << "Not addressable resource.";
            break;
        case UriProtocol::File:
            stream << "file" << Uri::getProtocolSeparator() << uri.getPath();
            break;
        case UriProtocol::Tcp:
            stream << "tcp" << Uri::getProtocolSeparator() << uri.getHost() << Uri::getPortSeparator() << uri.getPort()
                   << Uri::getPathSeparator() << uri.getPath();
            break;
        case UriProtocol::Udp:
            stream << "udp" << Uri::getProtocolSeparator() << uri.getHost() << Uri::getPortSeparator() << uri.getPort()
                   << Uri::getPathSeparator() << uri.getPath();
            break;
        };

        return stream;
    }

    //========================================
    //! \brief Compare two Uri's on equality.
    //! \param[in] luri  Left URI of operator.
    //! \param[in] luri  Right URI of operator.
    //! \returns Either \c true if both URI's exact equals, otherwise \c false.
    //----------------------------------------
    friend bool operator==(const Uri& lUri, const Uri& rUri)
    {
        return (lUri.getFormat() == rUri.getFormat()) && (lUri.getVersion() == rUri.getVersion())
               && (lUri.getSchema() == rUri.getSchema()) && (lUri.getProtocol() == rUri.getProtocol())
               && (lUri.getHost() == rUri.getHost()) && (lUri.getPort() == rUri.getPort())
               && (lUri.getPath() == rUri.getPath());
    }

    //========================================
    //! \brief Compare two Uri's on inequality.
    //! \param[in] luri  Left URI of operator.
    //! \param[in] luri  Right URI of operator.
    //! \returns Either \c true if one URI differ from the other, otherwise \c false.
    //----------------------------------------
    friend bool operator!=(const Uri& lUri, const Uri& rUri)
    {
        return (lUri.getFormat() != rUri.getFormat()) || (lUri.getVersion() != rUri.getVersion())
               || (lUri.getSchema() != rUri.getSchema()) || (lUri.getProtocol() != rUri.getProtocol())
               || (lUri.getHost() != rUri.getHost()) || (lUri.getPort() != rUri.getPort())
               || (lUri.getPath() != rUri.getPath());
    }

private:
    //========================================
    //! \brief Format like MDF, IDC etc.
    //----------------------------------------
    std::string m_format{};

    //========================================
    //! \brief Format version like 4.1.X etc.
    //----------------------------------------
    std::string m_version{};

    //========================================
    //! \brief Format schema like idc or pcap
    //!     in case of unrelated format like MDF.
    //----------------------------------------
    std::string m_schema{};

    //========================================
    //! \brief Uri transfer protocol.
    //----------------------------------------
    UriProtocol m_proto{UriProtocol::NoAddr};

    //========================================
    //! \brief Host name/ip identifier.
    //----------------------------------------
    std::string m_host{};

    //========================================
    //! \brief Port of the host system.
    //----------------------------------------
    uint16_t m_port{};

    //========================================
    //! \brief Path of the Host system.
    //----------------------------------------
    std::string m_path{};
};

//==============================================================================

//========================================
//! \brief Nullable Uri pointer.
//----------------------------------------
using UriPtr = std::unique_ptr<Uri>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
