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
//! \date Jul 13, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/IdcDataHeader.hpp>
#include <microvision/common/logging/logging.hpp>

#include <boost/asio.hpp>
#include <boost/optional.hpp>
#include <boost/asio/deadline_timer.hpp>
#include <boost/system/error_code.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief
//! \date Jul 12, 2016
//!
//! Class to receive idc messages from an TCP/IP socket and prepare them
//! for decoding.
//------------------------------------------------------------------------------
class MsgBufferBase
{
public:
    using ConstCharVectorPtr = std::shared_ptr<const std::vector<char>>;

    static constexpr int32_t maxBufCapacity = 1024 * 1024 * 4;

public:
    //========================================
    //! \brief Create buffer of given start size.
    //!
    //! \param[in]       bufSize     Size of the buffer
    //!                              which will be allocated
    //!                              to hold the received
    //!                              message data. This size has to be
    //!                              significantly larger than the largest
    //!                              to be expected message.
    //----------------------------------------
    explicit MsgBufferBase(const int32_t bufSize);

    virtual ~MsgBufferBase() = default;

public:
    //========================================
    //! \brief Return number of free bytes currently left in the buffer.
    //----------------------------------------
    int getBytesToRead() const { return this->m_bufCapacity - this->m_insAt; }

    //========================================
    //! \brief Return current receive buffer.
    //----------------------------------------
    boost::asio::mutable_buffers_1 getRecvBuf() const
    {
        return boost::asio::buffer(boost::asio::buffer(*this->m_bufPtr) + static_cast<std::size_t>(this->m_insAt));
    }

protected:
    //========================================
    //! \brief Process data in receive buffer.
    //!
    //! \param[out] dh                If a complete message is in the buffer,
    //!                               \a recvDataHeader will point to the attribute
    //!                               \a recvDataHeader which
    //!                               will contain the valid header data.
    //!                               Otherwise \a recvDataHeader will be \c NULL.
    //! \param[out] msgBody           If a complete message is in the buffer,
    //!                               \a startOfMsgBody will point to the start
    //!                               of the message body inside the objects buffer.
    //!                               Otherwise \a startOfMsgBody will be \c NULL.
    //! \param[in] nbOfBytesReceived  Number of bytes added to the buffer since last processing.
    //! \return \c True if a complete message is in the buffer, ready to be processed.
    //!         The message is described by \a recvDataHeader and \a startOfMsgBody.
    //!         \c false There is no complete message in the buffer yet. \a recvDataHeader
    //!         and \a startOfMsgBody will be NULL.
    //----------------------------------------
    bool processBuffer(const IdcDataHeader*& dh, ConstCharVectorPtr& msgBody, const int32_t nbOfBytesReceived);

    //========================================
    //! \brief Performs a buffer cleanup if necessary.
    //----------------------------------------
    void bufferCleanup();

    //========================================
    //! \brief Receive buffer will be completely cleaned.
    //----------------------------------------
    void clearBufferCompletely();

    //========================================
    //! \brief Receive buffer will be cleaned but up to 3 bytes will be preserved.
    //!
    //! That is the first 3 bytes might contain the first 3 bytes of the magic word.
    //----------------------------------------
    void clearBufferPreserve3Bytes();

    //========================================
    //! \brief Remove those bytes from the buffer which are scheduled for deletion.
    //----------------------------------------
    void removeScheduledBytes();

protected:
    //========================================
    //! \brief Reduce the position of the given counter \a pos by \a nbOfBytes.
    //!
    //! But \a lowerLimit is the minimal value for the new position of \a pos.
    //! \param[in,out] pos         On entry the current position of the counter.
    //!                            On exit the new by \a nbOfBytes reduced position,
    //!                            or \a lowerLimit if the now position would be less
    //!                            than \a lowerLimit.
    //! \param[in]     nbOfBytes   Number of bytes the counter has to be reduced by.
    //! \param[in]     lowerLimit  Minimal value the counter is allowed to have after
    //!                            the reduction.
    //----------------------------------------
    static void reducePos(int32_t& pos, const int32_t nbOfBytes, const int32_t lowerLimit = -1);

protected:
    static constexpr const char* loggerId = "microvision::common::sdk::MsgBufferBase";
    static microvision::common::logging::LoggerSPtr logger;

protected:
    size_t m_nbOfBytesRead;

    //========================================
    //! \brief Current size of the objects buffer #m_bufPtr.
    //----------------------------------------
    int32_t m_bufCapacity;

    //========================================
    //! \brief Pointer to the object's buffer.
    //----------------------------------------
    std::shared_ptr<std::vector<char>> m_bufPtr;

    //========================================
    //! \brief Position to insert new data.
    //!
    //! The buffer is filled up to position #m_insAt-1.
    //! #m_insAt is the first not used byte in the buffer.
    //! The value has to be between 0 and #m_bufCapacity-1.
    //----------------------------------------
    int32_t m_insAt;

    //========================================
    //! \brief Index of the last processed byte in the buffer.
    //!
    //! -1 means that no byte has been processed yet.
    //----------------------------------------
    int32_t m_processedTo;

    //========================================
    //! \brief Index up to which the data can be deleted inside the buffer.
    //!
    //! -1 means that no byte is there to be deleted.
    //----------------------------------------
    int32_t m_scheduledForDeletionTo;

    //========================================
    //! \brief Latest header data read from the message buffer.
    //----------------------------------------
    IdcDataHeader m_dh;

    //========================================
    //! \brief position of the latest header found in the message buffer.
    //----------------------------------------
    int32_t m_dhAt;

    //========================================
    //! \brief position of the body
    //!
    //! Has to be between -1 and #m_bufCapacity-1. A value
    //! of -1 means no body found in the current
    //! buffer.
    //----------------------------------------
    int32_t m_bodyAt;

    //========================================
    //! \brief Size of the (to be) received message body.
    //!
    //! Has to be >= 0. Set to -1 if no header has been
    //! read which contains the body size information.
    //----------------------------------------
    int32_t m_bodySize;

    //========================================
    //! \brief Number of bytes that are missing to complete
    //!        the message body.
    //!
    //! Set to -1 if buffer is empty/has been cleared.
    //----------------------------------------
    int32_t m_missingBytes;

    //========================================
    //! \brief Set if there is no not yet processed/reported
    //!        complete message in the buffer
    //----------------------------------------
    bool m_needToRecvData;
}; // MsgBufferBase

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
