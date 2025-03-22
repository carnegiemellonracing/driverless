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
#ifndef HS_LIDAR_ST_V3_H
#define HS_LIDAR_ST_V3_H

#include <udp_protocol_header.h>
#include "plat_utils.h"
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

struct HS_LIDAR_BODY_AZIMUTH_ST_V3 {
  uint16_t m_u16Azimuth;

  uint16_t GetAzimuth() const { return little_to_native(m_u16Azimuth); }

  void Print() const {
    printf("HS_LIDAR_BODY_AZIMUTH_ST_V3: azimuth:%u\n", GetAzimuth());
  }
} PACKED;

struct HS_LIDAR_BODY_FINE_AZIMUTH_ST_V3 {
  uint8_t m_u8FineAzimuth;

  uint8_t GetFineAzimuth() const { return m_u8FineAzimuth; }

  void Print() const {
    printf("HS_LIDAR_BODY_FINE_AZIMUTH_ST_V3: FineAzimuth:%u\n",
           GetFineAzimuth());
  }
} PACKED;

struct HS_LIDAR_BODY_CHN_NNIT_ST_V3 {
  uint16_t m_u16Distance;
  uint8_t m_u8Reflectivity;
  uint8_t m_u8Confidence;

  uint16_t GetDistance() const { return little_to_native(m_u16Distance); }
  uint8_t GetReflectivity() const { return m_u8Reflectivity; }
  uint8_t GetConfidenceLevel() const { return m_u8Confidence; }

  void Print() const {
    printf("HS_LIDAR_BODY_CHN_NNIT_ST_V3:\n");
    printf("Dist:%u, Reflectivity: %u, confidenceLevel:%d\n", GetDistance(),
           GetReflectivity(), GetConfidenceLevel());
  }
} PACKED;


struct HS_LIDAR_BODY_CRC_ST_V3 {
  uint32_t m_u32Crc;

  uint32_t GetCrc() const { return little_to_native(m_u32Crc); }

  void Print() const {
    printf("HS_LIDAR_BODY_CRC_ST_V3:\n");
    printf("crc:0x%08x\n", GetCrc());
  }
} PACKED;

struct HS_LIDAR_TAIL_ST_V3 {
  // shutdown flag, bit 0
  static const uint8_t kShutdown = 0x01;

  // return mode
  static const uint8_t kStrongestReturn = 0x37;
  static const uint8_t kLastReturn = 0x38;
  static const uint8_t kDualReturn = 0x39;

  ReservedInfo1 m_reservedInfo1;
  ReservedInfo2 m_reservedInfo2;
  uint8_t m_u8Shutdown;
  ReservedInfo3 m_reservedInfo3;
  uint8_t m_u8UReserved[8];
  int16_t m_i16MotorSpeed;
  uint32_t m_u32Timestamp;
  uint8_t m_u8ReturnMode;
  uint8_t m_u8FactoryInfo;
  uint8_t m_u8UTC[6];
  // uint32_t m_u32SeqNum;

  uint8_t GetStsID0() const { return m_reservedInfo1.GetID(); }
  uint16_t GetData0() const { return m_reservedInfo1.GetData(); }

  uint8_t GetStsID1() const { return m_reservedInfo2.GetID(); }
  uint16_t GetData1() const { return m_reservedInfo2.GetData(); }

  uint8_t HasShutdown() const { return m_u8Shutdown & kShutdown; }

  uint8_t GetStsID2() const { return m_reservedInfo3.GetID(); }
  uint16_t GetData2() const { return m_reservedInfo3.GetData(); }

  int16_t GetMotorSpeed() const { return little_to_native(m_i16MotorSpeed); }

  uint32_t GetTimestamp() const { return little_to_native(m_u32Timestamp); }

  uint8_t GetReturnMode() const { return m_u8ReturnMode; }
  bool IsLastReturn() const { return m_u8ReturnMode == kLastReturn; }
  bool IsStrongestReturn() const { return m_u8ReturnMode == kStrongestReturn; }
  bool IsDualReturn() const { return m_u8ReturnMode == kDualReturn; }

  uint8_t GetFactoryInfo() const { return m_u8FactoryInfo; }

  uint8_t GetUTCData(uint8_t index) const {
    return m_u8UTC[index < sizeof(m_u8UTC) ? index : 0];
  }
  uint64_t GetMicroLidarTimeU64() const {
    if (m_u8UTC[0] != 0) {
			struct tm t = {0};
			t.tm_year = m_u8UTC[0];
			if (t.tm_year >= 200) {
				t.tm_year -= 100;
			}
			t.tm_mon = m_u8UTC[1] - 1;
			t.tm_mday = m_u8UTC[2] + 1;
			t.tm_hour = m_u8UTC[3];
			t.tm_min = m_u8UTC[4];
			t.tm_sec = m_u8UTC[5];
			t.tm_isdst = 0;
#ifdef _MSC_VER
  TIME_ZONE_INFORMATION tzi;
  GetTimeZoneInformation(&tzi);
  long int timezone =  tzi.Bias * 60;
#endif
      return (mktime(&t) - timezone - 86400) * 1000000 + GetTimestamp() ;
		}
		else {
      uint32_t utc_time_big = *(uint32_t*)(&m_u8UTC[0] + 2);
      uint64_t unix_second = big_to_native(utc_time_big);
      return unix_second * 1000000 + GetTimestamp();
		}

  }

  // uint32_t GetSeqNum() const { return little_to_native(m_u32SeqNum); }
  // static uint32_t GetSeqNumSize() { return sizeof(m_u32SeqNum); }

  void Print() const {
    printf("HS_LIDAR_TAIL_ST_V3:\n");
    printf(
        "sts0:%d, data0:%d, sts1:%d, data1:%d, shutDown:%d, motorSpeed:%d, "
        "timestamp:%u, return_mode:0x%02x, factoryInfo:0x%02x, utc:%u %u "
        "%u %u %u %u\n",
        GetStsID0(), GetData0(), GetStsID1(), GetData1(), HasShutdown(),
        GetMotorSpeed(), GetTimestamp(), GetReturnMode(), GetFactoryInfo(),
        GetUTCData(0), GetUTCData(1), GetUTCData(2), GetUTCData(3),
        GetUTCData(4), GetUTCData(5));
  }

} PACKED;

struct HS_LIDAR_TAIL_SEQ_NUM_ST_V3 {
  uint32_t m_u32SeqNum;

  uint32_t GetSeqNum() const { return little_to_native(m_u32SeqNum); }
  static uint32_t GetSeqNumSize() { return sizeof(m_u32SeqNum); }

  void Print() const {
    printf("HS_LIDAR_TAIL_SEQ_NUM_ST_V3:\n");
    printf("seqNum: %u\n", GetSeqNum());
  }
} PACKED;

struct HS_LIDAR_TAIL_CRC_ST_V3 {
  uint32_t m_u32Crc;

  uint32_t GetCrc() const { return little_to_native(m_u32Crc); }

  void Print() const {
    printf("HS_LIDAR_TAIL_CRC_ST_V3:\n");
    printf("crc:0x%08x\n", GetCrc());
  }
} PACKED;

struct HS_LIDAR_CYBER_SECURITY_ST_V3 {
  uint8_t m_u8Signature[32];

  uint8_t GetSignatureData(uint8_t index) const {
    return m_u8Signature[index < sizeof(m_u8Signature) ? index : 0];
  }

  void Print() const {
    printf("HS_LIDAR_CYBER_SECURITY_ST_V3:\n");
    for (uint8_t i = 0; i < sizeof(m_u8Signature); i++)
      printf("Signature%d:%d, ", i, GetSignatureData(i));
    printf("\n");
  }
} PACKED;

struct HS_LIDAR_HEADER_ST_V3 {
  static const uint8_t kSequenceNum = 0x01;
  static const uint8_t kIMU = 0x02;
  static const uint8_t kFunctionSafety = 0x04;
  static const uint8_t kCyberSecurity = 0x08;
  static const uint8_t kConfidenceLevel = 0x10;

  static const uint8_t kDistUnit = 0x04;
  static const uint8_t kFirstBlockLastReturn = 0x01;
  static const uint8_t kFirstBlockStrongestReturn = 0x02;

  uint8_t m_u8LaserNum;
  uint8_t m_u8BlockNum;
  uint8_t m_u8EchoCount;
  uint8_t m_u8DistUnit;
  uint8_t m_u8EchoNum;
  uint8_t m_u8Status;

  uint8_t GetLaserNum() const { return m_u8LaserNum; }
  uint8_t GetBlockNum() const { return m_u8BlockNum; }
  double GetDistUnit() const { return m_u8DistUnit / 1000.f; }
  uint8_t GetEchoCount() const { return m_u8EchoCount; }

  bool IsFirstBlockLastReturn() const {
    return m_u8EchoCount == kFirstBlockLastReturn;
  }
  bool IsFirstBlockStrongestReturn() const {
    return m_u8EchoCount == kFirstBlockStrongestReturn;
  }
  uint8_t GetEchoNum() const { return m_u8EchoNum; }

  bool HasSeqNum() const { return m_u8Status & kSequenceNum; }
  bool HasIMU() const { return m_u8Status & kIMU; }
  bool HasFuncSafety() const { return m_u8Status & kFunctionSafety; }
  bool HasCyberSecurity() const { return m_u8Status & kCyberSecurity; }
  bool HasConfidenceLevel() const { return m_u8Status & kConfidenceLevel; }

  uint16_t GetPacketSize() const {
    return sizeof(HS_LIDAR_PRE_HEADER) + sizeof(HS_LIDAR_HEADER_ST_V3) +
           (sizeof(HS_LIDAR_BODY_AZIMUTH_ST_V3) +
            sizeof(HS_LIDAR_BODY_FINE_AZIMUTH_ST_V3) +
            sizeof(HS_LIDAR_BODY_CHN_NNIT_ST_V3) * GetLaserNum()) *
               GetBlockNum() +
           sizeof(HS_LIDAR_BODY_CRC_ST_V3) + sizeof(HS_LIDAR_TAIL_ST_V3) +
           (HasSeqNum() ? sizeof(HS_LIDAR_TAIL_SEQ_NUM_ST_V3) : 0) +
           sizeof(HS_LIDAR_TAIL_CRC_ST_V3) +
           (HasCyberSecurity() ? sizeof(HS_LIDAR_CYBER_SECURITY_ST_V3) : 0);
  }

  void Print() const {
    printf("HS_LIDAR_HEADER_ST_V3:\n");
    printf(
        "laserNum:%02u, block_num:%02u, DistUnit:%g, EchoCnt:%02u, "
        "EchoNum:%02u, HasSeqNum:%d, HasIMU:%d, "
        "HasFuncSafety:%d, HasCyberSecurity:%d, HasConfidence:%d\n",
        GetLaserNum(), GetBlockNum(), GetDistUnit(), GetEchoCount(),
        GetEchoNum(), HasSeqNum(), HasIMU(), HasFuncSafety(),
        HasCyberSecurity(), HasConfidenceLevel());
  }
} PACKED;


struct FaultMessageVersion4_3 {
 public:
  uint16_t sob;
  uint8_t version_info;
  uint8_t utc_time[6];
  uint32_t time_stamp;
  uint8_t operate_state;
  uint8_t fault_state;
  uint8_t fault_code_type;
  uint8_t rolling_counter;
  uint8_t total_fault_code_num;
  uint8_t fault_code_id;
  uint32_t fault_code;
  uint8_t time_division_multiplexing[27];
  uint8_t software_version[8];
  uint8_t heating_state;
  uint8_t lidar_high_temp_state;
  uint8_t reversed[3];
  uint32_t crc;
  uint8_t cycber_security[32];
  uint32_t GetTimestamp() const { return big_to_native(time_stamp); }
  uint32_t GetCrc() const { return little_to_native(crc); }
  uint32_t GetFaultCode() const { return big_to_native(fault_code); }
  uint64_t GetMicroLidarTimeU64() const {
    if (utc_time[0] != 0) {
			struct tm t = {0};
			t.tm_year = utc_time[0];
			if (t.tm_year >= 200) {
				t.tm_year -= 100;
			}
			t.tm_mon = utc_time[1] - 1;
			t.tm_mday = utc_time[2] + 1;
			t.tm_hour = utc_time[3];
			t.tm_min = utc_time[4];
			t.tm_sec = utc_time[5];
			t.tm_isdst = 0;
#ifdef _MSC_VER
  TIME_ZONE_INFORMATION tzi;
  GetTimeZoneInformation(&tzi);
  long int timezone =  tzi.Bias * 60;
#endif
      return (mktime(&t) - timezone - 86400) * 1000000 + GetTimestamp() ;
		}
		else {
      uint32_t utc_time_big = *(uint32_t*)(&utc_time[0] + 2);
      uint64_t unix_second = big_to_native(utc_time_big);
      return unix_second * 1000000 + GetTimestamp();
		}
  }
  void ParserLensDirtyState(
      LensDirtyState lens_dirty_state[LENS_AZIMUTH_AREA_NUM]
                                   [LENS_ELEVATION_AREA_NUM]) {
    for (int i = 0; i < LENS_AZIMUTH_AREA_NUM; i++) {
      uint16_t rawdata =
          (*((uint16_t *)(&time_division_multiplexing[3 + i * 2])));
      for (int j = 0; j < LENS_ELEVATION_AREA_NUM; j++) {
        uint16_t lens_dirty_state_temp =
            (rawdata << ((LENS_ELEVATION_AREA_NUM - j - 1) * 2));
        uint16_t lens_dirty_state_temp1 =
            (lens_dirty_state_temp >> ((LENS_ELEVATION_AREA_NUM - 1) * 2));
        if (time_division_multiplexing[0] == 1) {
          switch (lens_dirty_state_temp1) {
            case 0: {
              lens_dirty_state[i][j] = kLensNormal;
              break;
            }
            case 1: {
              lens_dirty_state[i][j] = kPassable;
              break;
            }
            case 3: {
              lens_dirty_state[i][j] = kUnPassable;
              break;
            }
            default:
              lens_dirty_state[i][j] = kUndefineData;
              break;
          }

        } else
          lens_dirty_state[i][j] = kUndefineData;
      }
    }
  }
  double ParserTemperature() {
    double temp =
        ((double)(*((uint16_t *)(&time_division_multiplexing[1])))) * 0.1f;
    return temp;
  }
  void ParserFaultMessage(FaultMessageInfo &fault_message_info) {
    fault_message_info.fault_prase_version = 0x43;
    fault_message_info.version = version_info;
    memcpy(fault_message_info.utc_time, utc_time, sizeof(utc_time));
    fault_message_info.timestamp = GetTimestamp();
    fault_message_info.total_time = static_cast<double>(GetMicroLidarTimeU64()) / 1000000.0;
    fault_message_info.operate_state = operate_state;
    fault_message_info.fault_state = fault_state;
    fault_message_info.total_faultcode_num = total_fault_code_num;
    fault_message_info.faultcode_id = fault_code_id;
    fault_message_info.faultcode = GetFaultCode();
    fault_message_info.union_info.fault4_3.fault_code_type = fault_code_type;
    fault_message_info.union_info.fault4_3.rolling_counter = rolling_counter;
    fault_message_info.union_info.fault4_3.tdm_data_indicate = time_division_multiplexing[0];
    memcpy(fault_message_info.union_info.fault4_3.time_division_multiplexing, time_division_multiplexing, sizeof(time_division_multiplexing));
    fault_message_info.union_info.fault4_3.software_id = *((uint16_t *)(&software_version[0]));
    fault_message_info.union_info.fault4_3.software_version = *((uint16_t *)(&software_version[2]));
    fault_message_info.union_info.fault4_3.hardware_version = *((uint16_t *)(&software_version[4]));
    fault_message_info.union_info.fault4_3.bt_version = *((uint16_t *)(&software_version[6]));
    fault_message_info.union_info.fault4_3.heating_state = heating_state;
    fault_message_info.union_info.fault4_3.high_temperture_shutdown_state = lidar_high_temp_state;
    memcpy(fault_message_info.union_info.fault4_3.reversed, reversed, sizeof(reversed));
  }
} PACKED;
#ifdef _MSC_VER
#pragma pack(pop)
#endif
}  // namespace lidar
}  // namespace hesai
#endif
