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
//! \date Sep 17, 2019
//------------------------------------------------------------------------------

//==============================================================================
// This class is based on the MD5 reference implementation of RFC 1321
// (see https://tools.ietf.org/html/rfc1321).
//------------------------------------------------------------------------------
//  Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
//  rights reserved.
//
//  License to copy and use this software is granted provided that it
//  is identified as the "RSA Data Security, Inc. MD5 Message-Digest
//  Algorithm" in all material mentioning or referencing this software
//  or this function.
//
//  License is also granted to make and use derivative works provided
//  that such works are identified as "derived from the RSA Data
//  Security, Inc. MD5 Message-Digest Algorithm" in all material
//  mentioning or referencing the derived work.
//
//  RSA Data Security, Inc. makes no representations concerning either
//  the merchantability of this software or the suitability of this
//  software for any particular purpose. It is provided "as is"
//  without express or implied warranty of any kind.
//
//  These notices must be retained in any copies of any part of this
//  documentation and/or software.
//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <array>
#include <cstring>
#include <iostream>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace crypto {
//==============================================================================

//==============================================================================
//! \brief This class implements a generator for MD5 checksums.
//!
//! The typical sequence is:
//! 1. create generator
//! 2. feed generator with input buffers
//! 3. finalize checksum generation
//! 4. get checksum
//! E.g.:
//! <code>
//! Md5 generator;
//! generator.update(input);
//! generator.update(input);
//! generator.finalize();
//! Digest digest = generator.getDigest();
//! </code>
//!
//! If the input is a single buffer only, the convenience constructor can be used instead:
//! <code>
//! Md5 generator{input};
//! Digest digest = generator.getDigest();
//! </code>
//------------------------------------------------------------------------------
class Md5
{
public:
    static constexpr uint32_t digestLength{16}; //!< Length of the digest created by this generator.
    using Digest = std::array<uint8_t, digestLength>; //!< Type of digest created by this generator.

public:
    //========================================
    //! \brief Constructor.
    //!
    //! Initializes the checksum generator.
    //----------------------------------------
    Md5();

    //========================================
    //! \brief Constructor.
    //!
    //! This constructor will create the digest immediately. It is not possible to add other buffers to the digest
    //! using one of the \a update methods.
    //!
    //! \param[in] input  Buffer containing the data to get the checksum for.
    //----------------------------------------
    explicit Md5(const std::vector<uint8_t>& input) : Md5(&input[0], input.size()) {}

    //========================================
    //! \brief Constructor.
    //!
    //! This constructor will create the digest immediately. It is not possible to add other buffers to the digest
    //! using one of the \a update methods.
    //!
    //! \param[in] input   Buffer containing the data to get the checksum for.
    //! \param[in] length  Size of the data in the buffer.
    //----------------------------------------
    explicit Md5(const uint8_t input[], const std::size_t length);

    //========================================
    //! \brief MD5 block update operation.
    //!
    //! Continues an MD5 message-digest operation by processing another message block
    //!
    //! \param[in] input  Buffer containing the data to get the checksum for.
    //----------------------------------------
    void update(const std::vector<uint8_t>& input) { update(&input[0], input.size()); }

    //========================================
    //! \brief MD5 block update operation.
    //!
    //! Continues an MD5 message-digest operation by processing another message block
    //!
    //! \param[in] input   Buffer containing the data to get the checksum for.
    //! \param[in] length  Size of the data in the buffer.
    //----------------------------------------
    void update(const uint8_t input[], const std::size_t length);

    //========================================
    //! \brief MD5 finalization.
    //! Ends an MD5 message-digest operation, writes the message digest, and clears the context.
    //----------------------------------------
    void finalize();

    //========================================
    //! \brief Get the message digest (checksum).
    //----------------------------------------
    Digest getDigest() const { return m_digest; }

private:
    static constexpr uint32_t blockSize{64};
    using Block = std::array<uint8_t, blockSize>;

    void init();

    void transform(const uint8_t block[], const std::size_t length);
    static void decode(uint32_t output[], const uint8_t input[], const std::size_t len);
    static void encode(uint8_t output[], const uint32_t input[], const std::size_t len);

    // low level logic operations
    static inline uint32_t rotateLeft(const uint32_t x, const uint32_t n);
    static inline uint32_t F(const uint32_t x, const uint32_t y, const uint32_t z);
    static inline uint32_t G(const uint32_t x, const uint32_t y, const uint32_t z);
    static inline uint32_t H(const uint32_t x, const uint32_t y, const uint32_t z);
    static inline uint32_t I(const uint32_t x, const uint32_t y, const uint32_t z);
    static inline void FF(uint32_t& a,
                          const uint32_t b,
                          const uint32_t c,
                          const uint32_t d,
                          const uint32_t x,
                          const uint32_t s,
                          const uint32_t ac);
    static inline void GG(uint32_t& a,
                          const uint32_t b,
                          const uint32_t c,
                          const uint32_t d,
                          const uint32_t x,
                          const uint32_t s,
                          const uint32_t ac);
    static inline void HH(uint32_t& a,
                          const uint32_t b,
                          const uint32_t c,
                          const uint32_t d,
                          const uint32_t x,
                          const uint32_t s,
                          const uint32_t ac);
    static inline void II(uint32_t& a,
                          const uint32_t b,
                          const uint32_t c,
                          const uint32_t d,
                          const uint32_t x,
                          const uint32_t s,
                          const uint32_t ac);

    bool m_finalized{false};
    Block m_buffer{};
    uint32_t m_count[2]{0, 0};
    uint32_t m_state[4]{0, 0, 0, 0};
    Digest m_digest{};
};

//==============================================================================
} // namespace crypto
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
