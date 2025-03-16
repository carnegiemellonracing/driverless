/************************************************************************************************
Copyright (C) 2023 Hesai Technology Co., Ltd.
Copyright (C) 2023 Original Authors
All rights reserved.

All code in this repository is released under the terms of the following Modified BSD License. 
Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and 
  the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and 
  the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
  promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
************************************************************************************************/

/*
 * File:       udp4_3_parser.h
 * Author:     Zhang Yu <zhangyu@hesaitech.com>
 * Description: Declare Udp4_3Parser class
*/

#ifndef UDP4_3_PARSER_H_
#define UDP4_3_PARSER_H_

#define CORRECTION_AZIMUTH_STEP (200)
#define CORRECTION_AZIMUTH_NUM (180)
#define AT128_LASER_NUM (128)
#define ANGULAR_RESOLUTION (256)
#define MARGINAL_ANGLE (7625) 
#define ACCEPTANCE_ANGLE (200)

#include <cmath>
#include "general_parser.h"
#include "lidar_types.h"
namespace hesai
{
namespace lidar
{
#ifdef _MSC_VER
#define PACKED
#pragma pack(push, 1)
#else
#define PACKED __attribute__((packed))
#endif
struct PandarATCorrectionsHeader {
  uint8_t delimiter[2];
  uint8_t version[2];
  uint8_t channel_number;
  uint8_t mirror_number;
  uint8_t frame_number;
  uint8_t frame_config[8];
  uint8_t resolution;
} PACKED;
static_assert(sizeof(PandarATCorrectionsHeader) == 16, "");

struct PandarATFrameInfo {
  uint32_t start_frame[8];
  uint32_t end_frame[8];
  int32_t azimuth[AT128_LASER_NUM];
  int32_t elevation[AT128_LASER_NUM];
  std::array<float, CIRCLE> sin_map;
  std::array<float, CIRCLE> cos_map;
};

struct PandarATCorrections {
 public:
  PandarATCorrectionsHeader header;
  uint16_t start_frame[8];
  uint16_t end_frame[8];
  int16_t azimuth[AT128_LASER_NUM];
  int16_t elevation[AT128_LASER_NUM];
  int8_t azimuth_offset[CIRCLE_ANGLE];
  int8_t elevation_offset[CIRCLE_ANGLE];
  uint8_t SHA256[32];
  PandarATFrameInfo l;  // V1.5
  PandarATCorrections()
  : header(), l()
  {
    memset(start_frame, 0, sizeof(start_frame));
    memset(end_frame, 0, sizeof(end_frame));
    memset(azimuth, 0, sizeof(azimuth));
    memset(elevation, 0, sizeof(elevation));
    memset(azimuth_offset, 0, sizeof(azimuth_offset));
    memset(elevation_offset, 0, sizeof(elevation_offset));
    memset(SHA256, 0, sizeof(SHA256));
  }

  static const int STEP3 = CORRECTION_AZIMUTH_STEP * kFineResolutionInt;
  float GetAzimuthAdjustV3(uint8_t ch, uint32_t azi) const {
    int i = int(std::floor(1.f * azi / STEP3));
    int l = azi - i * STEP3;
    float k = 1.f * l / STEP3;
    return round((1 - k) * azimuth_offset[ch * CORRECTION_AZIMUTH_NUM + i] +
                 k * azimuth_offset[ch * CORRECTION_AZIMUTH_NUM + i + 1]);
  }
  float GetElevationAdjustV3(uint8_t ch, uint32_t azi) const {
    int i = int(std::floor(1.f * azi / STEP3));
    int l = azi - i * STEP3;
    float k = 1.f * l / STEP3;
    return round((1 - k) * elevation_offset[ch * CORRECTION_AZIMUTH_NUM + i] +
                 k * elevation_offset[ch * CORRECTION_AZIMUTH_NUM + i + 1]);
  }
};
#ifdef _MSC_VER
#pragma pack(pop)
#endif
// class Udp4_3Parser
// parsers packets and computes points for PandarAT128
// you can parser the upd or pcap packets using the DocodePacket fuction
// you can compute xyzi of points using the ComputeXYZI fuction, which uses cpu to compute
template<typename T_Point>
class Udp4_3Parser : public GeneralParser<T_Point> {
 public:
  Udp4_3Parser();
  virtual ~Udp4_3Parser();                         
  // 从PandarATCorrections中获取
  int16_t GetVecticalAngle(int channel); 

  // determine whether frame splitting is needed
  bool IsNeedFrameSplit(uint16_t azimuth, int field);  

  // get lidar correction file from local file,and pass to udp parser                                 
  virtual void LoadCorrectionFile(std::string correction_path);
  virtual int LoadCorrectionString(char *correction_string);

  virtual void HandlePacketData(uint8_t *pu8Buf, uint16_t u16Len);
  // covert a origin udp packet to decoded data, and pass the decoded data to a frame struct to reduce memory copy   
  virtual int DecodePacket(LidarDecodedFrame<T_Point> &frame, const UdpPacket& udpPacket); 

  // compute xyzi of points from decoded packet
  // param packet is the decoded packet; xyzi of points after computed is puted in frame    
  virtual int ComputeXYZI(LidarDecodedFrame<T_Point> &frame, int packet_index);

  virtual void ParserFaultMessage(UdpPacket& udp_packet, FaultMessageInfo &fault_message_info);
  PandarATCorrections m_PandarAT_corrections;

 protected:
  int view_mode_;
  bool get_correction_file_;
};
}  // namespace lidar
}  // namespace hesai

#include "udp4_3_parser.cc"

#endif  // UDP4_3_PARSER_H_
