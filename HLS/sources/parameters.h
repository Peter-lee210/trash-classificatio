#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <stdint.h>
#include <ap_axi_sdata.h>
#include "D:/Tailieu/DA2/quant.h"

// ===================== BIT-WIDTH TỐI ƯU =====================
#define bit_act_CONV		18   // 9-bit * 8-bit + sign
#define bit_act_FC			16   // 8-bit * 7-bit + sign
#define bit_w_CONV			8
#define bit_w_FC			8
#define bit_i_CONV			3
#define bit_i_FC			4

// ===================== INDEX =====================
#define upper_act_CONV		17
#define upper_act_FC		15
#define upper_index_CONV	10
#define lower_index_CONV	7
#define upper_index_FC		11
#define lower_index_FC		7

// ===================== DATA TYPES =====================
typedef ap_int<bit_act_CONV> ACT_CONV;
typedef ap_int<bit_act_FC>   ACT_FC;
typedef ap_int<bit_w_CONV>   W_CONV;
typedef ap_int<bit_w_FC>     W_FC;
typedef ap_uint<bit_i_CONV>  I_CONV;
typedef ap_uint<bit_i_FC>    I_FC;
typedef int32_t              DATA_HW;
typedef int32_t              DATA_SW;
typedef ap_int<32>           EXT_DATA;
typedef ap_uint<8>           CALL_DATA;  // 0–255
typedef ap_axis<16, 2, 5, 6> DATA_STREAM; // AXIS data type

// ===================== TILE & LAYER SIZES =====================
#define tile_map          14
#define tile_conv_in      8     // giảm để tiết kiệm BRAM
#define tile_conv_out     8
#define tile_fc_in        16
#define tile_fc_out       32

#define in_map_LEN        (tile_map + 2)
#define out_map_LEN       tile_map
#define w_conv_LEN        (tile_conv_in * tile_conv_out * 3 / 2)
#define w_fc_LEN          (tile_fc_in * tile_fc_out / 2)
#define save_input_LEN    10240  // giảm để vừa BRAM

// ===================== CONSTANTS FOR HLS =====================
const int HLS_tile_map = tile_map;
const int HLS_tile_conv_in = tile_conv_in;
const int HLS_tile_conv_out = tile_conv_out;
const int HLS_in_map_LEN = in_map_LEN * in_map_LEN * tile_conv_out / 2;
const int HLS_out_map_LEN = tile_map * tile_map * tile_conv_out / 2;
const int HLS_residual_map_LEN = save_input_LEN;
const int HLS_w_conv_LEN = w_conv_LEN;
const int HLS_b_conv_LEN = tile_conv_out;
const int HLS_tile_fc_in = tile_fc_in;
const int HLS_tile_fc_out = tile_fc_out;
const int HLS_in_array_LEN = tile_fc_in;
const int HLS_out_array_LEN = tile_fc_out;
const int HLS_w_fc_LEN = w_fc_LEN;
const int HLS_b_fc_LEN = tile_fc_out;

// ===================== MEMORY LAYOUT FOR LAYERS =====================
const int map_size = 1572864; //(112/14)*(112/14)*16*16*96
const int array_size = 1280;
const int w_conv_layer = 1525656;
const int fc_layer = 360000;
const int b_conv_layer = 17056;
const int b_fc_layer = 1000;


// ===================== PE & PIPELINE =====================
#define MAX_CONV_3X3        32
#define MAX_CONVS           40
#define MAX_AVG             5
#define MAX_FC              64
#define number_PE           4
#define size_info           19
#define stage_pipeline      4

const DATA_HW MAX_CALL[5] = {
    MAX_CONV_3X3,  // type_layer 0
    MAX_CONVS,     // type_layer 1
    MAX_CONVS,     // type_layer 2
    MAX_AVG,       // type_layer 3
    MAX_FC         // type_layer 4
};

const int HLS_tile_size = 3 * MAX_FC / 2;
const int HLS_info_size = size_info * MAX_FC / 2;

const DATA_HW zeros_for_CONV_PE = 6;
const DATA_HW zeros_for_FC_PE = 8;

// ===================== QUANTIZATION TABLES =====================
// [layer][inter_layer][0:act,1:bias,2:relu,3:res]
const DATA_HW CONV_quant[19][3][4] = {
    {{act_conv_0, bias_conv_0, relu_conv_0, res_conv_0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
    {{act_conv_1, bias_conv_1, relu_conv_1, res_conv_1}, {act_conv_2, bias_conv_2, relu_conv_2, res_conv_2}, {0, 0, 0, 0}},
    {{act_conv_3, bias_conv_3, relu_conv_3, res_conv_3}, {act_conv_4, bias_conv_4, relu_conv_4, res_conv_4}, {act_conv_5, bias_conv_5, relu_conv_5, res_conv_5}},
    {{act_conv_6, bias_conv_6, relu_conv_6, res_conv_6}, {act_conv_7, bias_conv_7, relu_conv_7, res_conv_7}, {act_conv_8, bias_conv_8, relu_conv_8, res_conv_8}},
    {{act_conv_9, bias_conv_9, relu_conv_9, res_conv_9}, {act_conv_10, bias_conv_10, relu_conv_10, res_conv_10}, {act_conv_11, bias_conv_11, relu_conv_11, res_conv_11}},
    {{act_conv_12, bias_conv_12, relu_conv_12, res_conv_12}, {act_conv_13, bias_conv_13, relu_conv_13, res_conv_13}, {act_conv_14, bias_conv_14, relu_conv_14, res_conv_14}},
    {{act_conv_15, bias_conv_15, relu_conv_15, res_conv_15}, {act_conv_16, bias_conv_16, relu_conv_16, res_conv_16}, {act_conv_17, bias_conv_17, relu_conv_17, res_conv_17}},
    {{act_conv_18, bias_conv_18, relu_conv_18, res_conv_18}, {act_conv_19, bias_conv_19, relu_conv_19, res_conv_19}, {act_conv_20, bias_conv_20, relu_conv_20, res_conv_20}},
    {{act_conv_21, bias_conv_21, relu_conv_21, res_conv_21}, {act_conv_22, bias_conv_22, relu_conv_22, res_conv_22}, {act_conv_23, bias_conv_23, relu_conv_23, res_conv_23}},
    {{act_conv_24, bias_conv_24, relu_conv_24, res_conv_24}, {act_conv_25, bias_conv_25, relu_conv_25, res_conv_25}, {act_conv_26, bias_conv_26, relu_conv_26, res_conv_26}},
    {{act_conv_27, bias_conv_27, relu_conv_27, res_conv_27}, {act_conv_28, bias_conv_28, relu_conv_28, res_conv_28}, {act_conv_29, bias_conv_29, relu_conv_29, res_conv_29}},
    {{act_conv_30, bias_conv_30, relu_conv_30, res_conv_30}, {act_conv_31, bias_conv_31, relu_conv_31, res_conv_31}, {act_conv_32, bias_conv_32, relu_conv_32, res_conv_32}},
    {{act_conv_33, bias_conv_33, relu_conv_33, res_conv_33}, {act_conv_34, bias_conv_34, relu_conv_34, res_conv_34}, {act_conv_35, bias_conv_35, relu_conv_35, res_conv_35}},
    {{act_conv_36, bias_conv_36, relu_conv_36, res_conv_36}, {act_conv_37, bias_conv_37, relu_conv_37, res_conv_37}, {act_conv_38, bias_conv_38, relu_conv_38, res_conv_38}},
    {{act_conv_39, bias_conv_39, relu_conv_39, res_conv_39}, {act_conv_40, bias_conv_40, relu_conv_40, res_conv_40}, {act_conv_41, bias_conv_41, relu_conv_41, res_conv_41}},
    {{act_conv_42, bias_conv_42, relu_conv_42, res_conv_42}, {act_conv_43, bias_conv_43, relu_conv_43, res_conv_43}, {act_conv_44, bias_conv_44, relu_conv_44, res_conv_44}},
    {{act_conv_45, bias_conv_45, relu_conv_45, res_conv_45}, {act_conv_46, bias_conv_46, relu_conv_46, res_conv_46}, {act_conv_47, bias_conv_47, relu_conv_47, res_conv_47}},
    {{act_conv_48, bias_conv_48, relu_conv_48, res_conv_48}, {act_conv_49, bias_conv_49, relu_conv_49, res_conv_49}, {act_conv_50, bias_conv_50, relu_conv_50, res_conv_50}},
    {{act_conv_51, bias_conv_51, relu_conv_51, res_conv_51}, {0, 0, 0, 0}, {0, 0, 0, 0}},
};

const DATA_HW AVG_quant = avg_quant;
const DATA_HW FC_quant[2] = {act_fc_quant, bias_fc_quant};

#endif
