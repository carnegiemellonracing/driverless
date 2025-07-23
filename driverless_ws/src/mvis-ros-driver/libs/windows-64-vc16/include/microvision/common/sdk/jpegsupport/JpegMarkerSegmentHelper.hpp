//==============================================================================
//!\file
//!
//! \brief Jpeg read helper. See \link https://datatracker.ietf.org/doc/rfc2035/.
//!
//! \date Apr 28th, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include "microvision/common/sdk/misc/defines/defines.hpp"

#include <cstring>
#include <cstdint>

//==============================================================================
//! \brief Functionality for reading and storing jpeg markers in RTP protocol.
//==============================================================================
class JpegMarkerSegmentHelper
{
private:
    static const uint8_t lum_dc_codelens[];

    static const uint8_t lum_dc_symbols[];

    static const uint8_t lum_ac_codelens[];

    static const uint8_t lum_ac_symbols[];

    static const uint8_t chm_dc_codelens[];

    static const uint8_t chm_dc_symbols[];

    static const uint8_t chm_ac_codelens[];

    static const uint8_t chm_ac_symbols[];

private:
    static uint8_t* makeQuantHeader(uint8_t* p, const uint8_t* qt, int tableNo);

    static uint8_t* makeHuffmanHeader(uint8_t* p,
                                      uint8_t* codelens,
                                      int ncodes,
                                      uint8_t* symbols,
                                      int nsymbols,
                                      int tableNo,
                                      int tableClass);

    static uint8_t* makeDRIHeader(uint8_t* p, uint16_t dri);

public:
    //!
    //! Arguments:
    //!   type, width, height: as supplied in RTP/JPEG header
    //!   lqt, cqt: quantization tables as either derived from
    //!        the Q field using MakeTables() or as specified
    //!        in section 4.2.
    //!   dri: restart interval in MCUs, or 0 if no restarts.
    //!
    //!   p: pointer to return area
    //!
    //! Return value:
    //!   The length of the generated headers.
    //!
    //!   Generate a frame and scan headers that can be prepended to the
    //!   RTP/JPEG data payload to produce a JPEG compressed image in
    //!   interchange format (except for possible trailing garbage and
    //!   absence of an EOI marker to terminate the scan).
    //!
    static int makeHeaders(uint8_t* p, int type, int w, int h, const uint8_t* lqt, const uint8_t* cqt, uint16_t dri);
};

//==============================================================================
