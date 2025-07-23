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
//! \date Mar 05, 2020
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/SharedBuffer.hpp>

#include <microvision/common/logging/logging.hpp>

#include <iostream>
#include <list>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief std::streambuf implementation to access multiple SharedBuffer as one stream.
//! \extends std::streambuf
//------------------------------------------------------------------------------
class SharedBufferStreamBuffer final : public std::streambuf
{
private:
    //========================================
    //! \brief Logger name to setup configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::SharedBufferStreamBuffer"};

    //========================================
    //! \brief Provides the common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

    //========================================
    //! \brief Buffer cache data for buffer list.
    //----------------------------------------
    struct Buffer
    {
        SharedBuffer buffer; //!< Buffer handle
        std::size_t position; //!< Buffer offset position
    };

    //========================================
    //! \brief List type for buffers.
    //----------------------------------------
    using BufferList = std::list<Buffer>;

public:
    //========================================
    //! \brief Construct from list of \c SharedBuffer objects.
    //! \param[in] buffers  List of \c SharedBuffer objects.
    //----------------------------------------
    SharedBufferStreamBuffer(const std::initializer_list<SharedBuffer>& buffers);

    //========================================
    //! \brief Construct from list of \c SharedBuffer objects.
    //! \param[in] buffers  List of \c SharedBuffer objects.
    //----------------------------------------
    SharedBufferStreamBuffer(const std::list<SharedBuffer>& buffers);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~SharedBufferStreamBuffer() override = default;

public:
    //========================================
    //! \brief Get the size of all buffers.
    //! \return Size of the stream.
    //----------------------------------------
    std::streamsize getStreamSize() const;

private: // positioning
    //========================================
    //! \brief Alters the stream positions.
    //!
    //! Each derived class provides its own appropriate behavior.
    //!
    //! \note Base class version does nothing, returns a \c pos_type
    //!       that represents an invalid stream position.
    //!
    //! \param[in] pos      Absolute position to set the position indicator to.
    //! \param[in] which    Defines which of the input and/or output sequences to affect.
    //! \return The resulting absolute position as defined by the position indicator.
    //!          Or if not reachable eof().
    //----------------------------------------
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;

    //========================================
    //! \brief Alters the stream positions.
    //!
    //! Each derived class provides its own appropriate behavior.
    //!
    //! \note Base class version does nothing, returns a \c pos_type
    //!       that represents an invalid stream position.
    //!
    //! \param[in] off      Relative position to set the position indicator to.
    //! \param[in] dir      Defines base position to apply the relative offset to.
    //! \param[in] which    Defines which of the input and/or output sequences to affect.
    //! \return The resulting absolute position as defined by the position indicator.
    //!          Or if not reachable eof().
    //----------------------------------------
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override;

private: // get area
    //========================================
    //! \brief Fetches more data from the controlled sequence.
    //!
    //! Informally, this function is called when the input buffer is
    //! exhausted (or does not exist, as buffering need not actually be
    //! done).  If a buffer exists, it is \a refilled.  In either case, the
    //! next available character is returned, or \c traits::eof() to
    //! indicate a null pending sequence.
    //!
    //! For a formal definition of the pending sequence, see a good text
    //! such as Langer & Kreft, or [27.5.2.4.3]/7-14.
    //!
    //! A functioning input streambuf can be created by overriding only
    //! this function (no buffer area will be used).  For an example, see
    //! https://gcc.gnu.org/onlinedocs/libstdc++/manual/streambufs.html
    //!
    //! \note Base class version does nothing, returns eof().
    //!
    //! \return The first character from the <em>pending sequence</em>.
    //----------------------------------------
    int_type underflow() override;

private: // put area
    //========================================
    //! \brief Consumes data from the buffer; writes to the controlled sequence.
    //!
    //! Informally, this function is called when the output buffer
    //! is full (or does not exist, as buffering need not actually
    //! be done).  If a buffer exists, it is \a consumed, with
    //! <em>some effect</em> on the controlled sequence.
    //! (Typically, the buffer is written out to the sequence
    //! verbatim.)  In either case, the character \a c is also
    //! written out, if \a __c is not \c eof().
    //!
    //! For a formal definition of this function, see a good text
    //! such as Langer & Kreft, or [27.5.2.4.5]/3-7.
    //!
    //! A functioning output streambuf can be created by overriding only
    //! this function (no buffer area will be used).
    //!
    //! \note Base class version does nothing, returns eof().
    //!
    //! \param[in] ch  An additional character to consume.
    //! \return eof() to indicate failure, something else (usually \a ch, or not_eof())
    //----------------------------------------
    int_type overflow(int_type ch) override;

private: // putback
    //========================================
    //! \brief Tries to back up the input sequence.
    //! \param[in] ch  The character to be inserted back into the sequence.
    //! \return eof() on failure, <em>some other value</em> on success
    //! \post The constraints of \c gptr(), \c eback(), and \c pptr()
    //!       are the same as for \c underflow().
    //!
    //! \note Base class version does nothing, returns eof().
    //----------------------------------------
    int_type pbackfail(int_type ch) override;

private:
    //========================================
    //! \brief List of all buffers.
    //----------------------------------------
    BufferList m_buffers;

    //========================================
    //! \brief Iterator on current get buffer.
    //----------------------------------------
    typename BufferList::iterator m_getBuffer;

    //========================================
    //! \brief Iterator on current put buffer.
    //----------------------------------------
    typename BufferList::iterator m_putBuffer;

}; // class SharedBufferStreamBuffer

//==============================================================================
//! \brief std::iostream implementation to access multiple SharedBuffer as one stream.
//! \extends std::iostream
//------------------------------------------------------------------------------
class SharedBufferStream final : public std::iostream
{
public:
    //========================================
    //! \brief List type for buffers.
    //----------------------------------------
    using BufferList = std::list<SharedBuffer>;

public:
    //========================================
    //! \brief Construct from list of \c SharedBuffer objects.
    //! \param[in] buffers  List of \c SharedBuffer objects.
    //----------------------------------------
    SharedBufferStream(const std::initializer_list<SharedBuffer>& buffers);

    //========================================
    //! \brief Construct from list of \c SharedBuffer objects.
    //! \param[in] buffers  List of \c SharedBuffer objects.
    //----------------------------------------
    SharedBufferStream(const std::list<SharedBuffer>& buffers);

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~SharedBufferStream() override = default;

public:
    //========================================
    //! \brief Get the size of all buffers.
    //! \return Size of the stream.
    //----------------------------------------
    std::streamsize getStreamSize() const;

private:
    //========================================
    //! \brief Stream buffer
    //----------------------------------------
    SharedBufferStreamBuffer m_buffer;

}; // class SharedBufferStream

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
