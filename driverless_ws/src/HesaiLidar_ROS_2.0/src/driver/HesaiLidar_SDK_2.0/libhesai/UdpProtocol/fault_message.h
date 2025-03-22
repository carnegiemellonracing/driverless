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
#ifndef FAULT_MESSAGE_H
#define FAULT_MESSAGE_H

#define LENS_AZIMUTH_AREA_NUM (12)
#define LENS_ELEVATION_AREA_NUM (8)

enum LensDirtyState {
  kUndefineData = -1,
  kLensNormal = 0,
  kPassable = 1,
  kUnPassable = 3,
};

struct FaultMessageInfo4_3 {
  uint8_t fault_code_type;
  uint8_t rolling_counter;
  uint8_t tdm_data_indicate;
  uint8_t time_division_multiplexing[27];
  uint16_t software_id;
  uint16_t software_version;
  uint16_t hardware_version;
  uint16_t bt_version;
  uint8_t heating_state;
  uint8_t high_temperture_shutdown_state;
  uint8_t reversed[3];
  void Print() const {
    printf("faultcode_type: %d\n", fault_code_type);
    printf("rolling_counter: %u\n", rolling_counter);
    printf("tdm_data_indicate: %d\n", tdm_data_indicate);
    printf("tdm_data:");
    for (int i = 0; i < 27; i++) {
      printf(" 0x%02x", time_division_multiplexing[i]);
    }
    printf("\n");
    printf("software_id: %04x, software_version: %04x, hardware_version: %04x, bt_version: %04x\n", 
            software_id, software_version, hardware_version, bt_version);
    printf("heating_state: %d\n", heating_state);
    printf("lidar_high_temp_state: %d\n", high_temperture_shutdown_state);
  }
};

struct FaultMessageInfo4_7 {
  uint8_t tdm_data_indicate;
  uint8_t time_division_multiplexing[14];
  uint8_t internal_fault_id;
  uint8_t fault_indicate[8];
  uint8_t customer_id;
  uint8_t software_version;
  uint8_t iteration_version;
  uint8_t reversed[17];
  void Print() const {
    printf("tdm_data_indicate: %d\n", tdm_data_indicate);
    printf("tdm_data:");
    for (int i = 0; i < 14; i++) {
      printf(" 0x%02x", time_division_multiplexing[i]);
    }
    printf("\n");
    printf("internal_fault_id: %d\n", internal_fault_id);
    printf("tdm_data:");
    for (int i = 0; i < 8; i++) {
      printf(" 0x%02x", fault_indicate[i]);
    }
    printf("\n");
    printf("customer_id: %02x, software_version: %02x, iteration_version: %02x\n", 
            customer_id, software_version, iteration_version);
  }
};

union FaultMessageUnionInfo {
  FaultMessageInfo4_3 fault4_3;
  FaultMessageInfo4_7 fault4_7;
};

struct FaultMessageInfo {
  uint8_t fault_prase_version;
  uint8_t version;
  uint8_t utc_time[6];
  uint32_t timestamp;
  double total_time;
  uint8_t operate_state;
  uint8_t fault_state;
  uint8_t total_faultcode_num;
  uint8_t faultcode_id;
  uint32_t faultcode;
  FaultMessageUnionInfo union_info;
  void Print() const {
    printf("version: %u\n", version);
    printf("utc_time: %u.%u.%u %u:%u:%u.%u\n", utc_time[0], utc_time[1], 
            utc_time[2], utc_time[3], utc_time[4], utc_time[5], timestamp);
    printf("total_time: %lf\n", total_time);
    printf("operate_state: %u\n", operate_state);
    printf("fault_state: %d\n", fault_state);
    printf("total_faultcode_num: %d, faultcode_id: %d, faultcode: 0x%08x\n", 
            total_faultcode_num, faultcode_id, faultcode);
    switch (fault_prase_version) {
      case 0x43:
        union_info.fault4_3.Print();
        break;
      case 0x47:
        union_info.fault4_7.Print();
        break;
      default:
        break;
    }
  }
};

#endif