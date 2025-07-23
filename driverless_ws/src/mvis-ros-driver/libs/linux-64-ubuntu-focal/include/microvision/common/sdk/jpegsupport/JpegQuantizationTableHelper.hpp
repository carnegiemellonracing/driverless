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

#include <cstdint>

//==============================================================================

//==============================================================================
//! \brief Functionality for reading and storing jpeg quantization tables used in RTP protocol jpegs.
//==============================================================================
class JpegQuantizationTableHelper
{
public:
    static constexpr uint16_t quantizationTableSize{64}; //!< size of jpeg one of luma and chroma quantisation tables
    static constexpr uint16_t chromaLumaQuantizationTableSize{
        quantizationTableSize * 2}; //!< size of jpeg both luma and chroma quantisation tables

private:
    static const uint8_t jpeg_luma_quantizer[quantizationTableSize];

    static const uint8_t jpeg_chroma_quantizer[quantizationTableSize];

public:
    //! Call MakeTables with the Q factor and two uint8_t[64] return arrays
    static void makeTables(int q, uint8_t* lqt, uint8_t* cqt);
};

//==============================================================================
