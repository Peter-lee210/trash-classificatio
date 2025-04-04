#include <hls_stream.h>
#include "D:/Tailieu/DA2/MobilenetV2_HLS/parameters.h"
#include <ap_axi_sdata.h>
// HARDWARE FUNCTIONS
void layer_CONV_3x3(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
                    volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                    DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer);

void layer_expansion_projection(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
                                hls::stream<DATA_STREAM> &ext_residual_map_read, hls::stream<DATA_STREAM> &ext_residual_map_write,
                                volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                                DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer);

void layer_depthwise(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
                     volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                     DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer);

void layer_AVG(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
               DATA_HW tile_index[3], DATA_HW info[size_info], DATA_HW quant[3], DATA_HW type_layer);

void layer_FC(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
              volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
              DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer);

ACT_CONV convolution_1x1(ACT_CONV in_map, W_CONV kernel);
ACT_CONV ReLU6(ACT_CONV in_map, DATA_HW upper);
DATA_HW MIN(DATA_HW x, DATA_HW y);

void read_w_conv(volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                 W_CONV w_conv[w_conv_LEN], I_CONV i_conv[w_conv_LEN], ACT_CONV b_conv[tile_conv_out],
                 DATA_HW tile_index[3], DATA_HW info[size_info]);

void read_w_fc(volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
               W_FC w_fc[w_fc_LEN], I_FC i_fc[w_fc_LEN], ACT_FC b_fc[tile_fc_out],
               DATA_HW LEN_W, DATA_HW LEN_B, DATA_HW info[size_info], DATA_HW type_layer);

void generate_quant(DATA_SW layer, DATA_SW inter_layer, DATA_SW type_layer,
                    DATA_HW quant[4]);

void generate_info_tile(volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
                        DATA_HW tile_index[3], DATA_HW info[size_info],
                        CALL_DATA call_PE, DATA_HW type_layer);

void generate_tile(volatile DATA_SW *ext_tile,
                   DATA_HW tile_index[3],
                   CALL_DATA call_PE, DATA_HW type_layer);

void generate_info(volatile DATA_SW *ext_info,
                   DATA_HW info[size_info],
                   CALL_DATA call_PE, DATA_HW type_layer);

void PEs(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
         hls::stream<DATA_STREAM> &ext_residual_map_read, hls::stream<DATA_STREAM> &ext_residual_map_write,
         volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
         volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
         DATA_HW tile_index[3], DATA_SW quant[4], DATA_SW info[size_info], CALL_DATA call_PE, DATA_SW type_layer);

void MobileNet_Stream(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data, hls::stream<DATA_STREAM> &ext_residual_map_write, hls::stream<DATA_STREAM> &ext_residual_map_read,
                     volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                     volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
                     volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
                     DATA_HW layer, DATA_SW inter_layer, DATA_SW type_layer);


// Helper functions to handle reading and writing streams
void read_data_stream(hls::stream<DATA_STREAM> &ext_in_data, DATA_STREAM &valIn_data) {
    #pragma HLS INLINE
    valIn_data = ext_in_data.read();  // Read data from stream into valIn_data
}

// Function to write data into the stream
void write_data_stream(hls::stream<DATA_STREAM> &ext_out_data, DATA_STREAM &valOut_data) {
    #pragma HLS INLINE
    ext_out_data.write(valOut_data);  // Write data from valOut_data to ext_out_data
}

void layer_CONV_3x3(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
                    volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                    DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer) {
    ACT_CONV line_buffer[2][in_map_LEN];
    ACT_CONV window[3][3], reg[3];
    bool stride_x = (info[3] == 1), stride_y = (info[3] == 1);
    W_CONV kernel[9];
    ACT_CONV temp, bias_value;
    DATA_HW limit = info[1]*info[1];

    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=window complete
    #pragma HLS ARRAY_PARTITION variable=reg complete
    #pragma HLS ARRAY_PARTITION variable=kernel complete

    ACT_CONV temp_map[in_map_LEN][in_map_LEN];
    DATA_STREAM valIn_data, valOut_data;
    DATA_HW index = 0;
    DATA_HW len = info[0]*info[0]*tile_index[0];

    // === Bọc truy cập bias ban đầu ===
    DATA_HW bias_index = info[14];
    if (bias_index >= b_conv_layer) {
        printf("[ERROR] Bias index out of bound: %d >= %d\n", bias_index, b_conv_layer);
        exit(1);
    }
    bias_value = (ACT_CONV)((EXT_DATA) ext_b_conv[bias_index]).range(upper_act_CONV, 0);

    LOOP_depth_3X3: for (DATA_HW i_tile = 0, j_tile = 0; i_tile < tile_index[0]; j_tile++) {
        // === Load kernel an toàn ===
        LOOP_kernel_3X3: for (DATA_HW k = 0; k < 9; ++k) {
            #pragma HLS PIPELINE II=1
            DATA_HW kernel_index = j_tile * 9 + i_tile * tile_index[1] * 9 + k + info[15];
            if (kernel_index >= w_conv_layer) {
                printf("[ERROR] Weight index out of bound: %d >= %d\n", kernel_index, w_conv_layer);
                exit(1);
            }
            kernel[k] = ((EXT_DATA) ext_w_conv[kernel_index]).range(lower_index_CONV, 0);
        }

        // Read line_buffer initial values
        DATA_HW read_count = info[1] + 2;
        for (DATA_HW l_tile = info[1] - 2; l_tile < info[1]; l_tile++) {
            valIn_data = ext_in_data.read();
            line_buffer[0][l_tile] = (ACT_CONV)(valIn_data.data >> quant[0]);
        }
        for (DATA_HW l_tile = 0; l_tile < info[1]; l_tile++) {
            valIn_data = ext_in_data.read();
            line_buffer[1][l_tile] = (ACT_CONV)(valIn_data.data >> quant[0]);
        }

        // Initialize window
        for (int x = 1; x < 3; x++) {
            for (int y = 1; y < 3; y++) {
                window[x][y] = line_buffer[x - 1][y + info[1] - 3];
            }
        }

        // Process map
        for (DATA_HW k_tile = 0; k_tile < info[1]; k_tile++) {
            for (DATA_HW l_tile = 0; l_tile < info[1]; l_tile++) {
                if (k_tile > 0 && k_tile < info[10] && l_tile > 0 && l_tile < info[10] && stride_x && stride_y) {
                    temp = 0;
                    for (int x = 0; x < 3; x++) {
                        for (int y = 0; y < 3; y++) {
                            temp += window[x][y] * kernel[x * 3 + y];
                        }
                    }

                    if (info[2] && j_tile == 0 && type_layer == 0) {
                        temp_map[k_tile][l_tile] = temp;
                    } else if (info[4] && j_tile == info[5]) {
                        temp += bias_value + temp_map[k_tile][l_tile];
                        temp = ReLU6(temp, quant[2]);

                        valOut_data.data = (DATA_HW) temp;
                        valOut_data.keep = valIn_data.keep;
                        valOut_data.strb = valIn_data.strb;
                        valOut_data.user = valIn_data.user;
                        valOut_data.id = valIn_data.id;
                        valOut_data.dest = valIn_data.dest;
                        valOut_data.last = (info[16] && index + 1 == len) ? 1 : 0;
                        ext_out_data.write(valOut_data);
                        index++;
                    } else {
                        temp_map[k_tile][l_tile] += temp;
                    }
                }

                // Shift line_buffer
                reg[0] = line_buffer[0][l_tile];
                reg[1] = line_buffer[0][l_tile] = line_buffer[1][l_tile];
                if (read_count < limit) {
                    valIn_data = ext_in_data.read();
                    read_count++;
                }
                reg[2] = line_buffer[1][l_tile] = (ACT_CONV)(valIn_data.data >> quant[0]);

                // Shift window
                for (int x = 0; x < 3; x++) {
                    window[x][0] = window[x][1];
                    window[x][1] = window[x][2];
                    window[x][2] = reg[x];
                }

                if (info[3] == 2) stride_x = !stride_x;
            }

            if (info[3] == 2) stride_y = !stride_y;
        }

        // Bước tile tiếp theo
        if (j_tile + 1 >= tile_index[1]) {
            j_tile = -1;
            i_tile++;
            DATA_HW bias_index_next = i_tile + info[14];
            if (bias_index_next >= b_conv_layer) {
                printf("[ERROR] Bias index out of bound: %d >= %d\n", bias_index_next, b_conv_layer);
                exit(1);
            }
            bias_value = (ACT_CONV)((EXT_DATA) ext_b_conv[bias_index_next]).range(upper_act_CONV, 0);
        }
    }

    // Đảm bảo stream kết thúc
    if (!ext_out_data.empty()) {
        DATA_STREAM temp;
        do {
            #pragma HLS pipeline II=1
            temp = ext_out_data.read();
        } while (!temp.last);
    }
}

void layer_expansion_projection(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
                                hls::stream<DATA_STREAM> &ext_residual_map_read, hls::stream<DATA_STREAM> &ext_residual_map_write,
                                volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                                DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer) {

    // Tối ưu bộ nhớ
    static ACT_CONV temp_map[tile_conv_out][out_map_LEN][out_map_LEN];
    ACT_CONV in_map[tile_conv_out][out_map_LEN][out_map_LEN];
    ACT_CONV temp;   // convolution
    W_CONV kernel_value;   // weight value
    ACT_CONV bias_value;   // bias
    ACT_CONV res_map_data;   // residual map data
    DATA_STREAM valIn_data, valOut_data, valIn_res, valOut_res;   // stream
    DATA_HW len = tile_index[2] * tile_index[2] * tile_index[0], read_res_info = 1, index = 0;
    DATA_HW read_data[tile_conv_in] = {0};   // read data

    #pragma HLS ARRAY_PARTITION variable=temp_map dim=2 cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=in_map dim=2 cyclic factor=8
    #pragma HLS RESOURCE variable=temp_map core=RAM_1P_BRAM
    #pragma HLS RESOURCE variable=in_map core=RAM_1P_BRAM
    #pragma HLS DATAFLOW
    #pragma HLS STREAM variable=ext_residual_map_read depth=512
    #pragma HLS STREAM variable=ext_residual_map_write depth=512
    #pragma HLS STREAM variable=ext_in_data depth=512
    #pragma HLS STREAM variable=ext_out_data depth=512

    // read weights
    W_CONV w_conv[w_conv_LEN];
    ACT_CONV b_conv[tile_conv_out];
    I_CONV i_conv[w_conv_LEN];
    #pragma HLS RESOURCE variable=w_conv core=RAM_1P_BRAM
    #pragma HLS RESOURCE variable=b_conv core=RAM_1P_LUTRAM
    #pragma HLS RESOURCE variable=i_conv core=XPM_MEMORY uram

    read_w_conv(ext_w_conv, ext_b_conv, w_conv, i_conv, b_conv, tile_index, info);
    bias_value = b_conv[0];   // bias

    LOOP_depth_expproj: for (DATA_HW i_tile = 0, j_tile = 0, pos_weight = 0, distance = 0; i_tile < tile_index[0]; j_tile++) {
        #pragma HLS LOOP_TRIPCOUNT min=96 max=1024
        #pragma HLS loop_flatten off

        if (distance == i_conv[pos_weight]) {
            kernel_value = w_conv[pos_weight];

            // Loop through map
            LOOP_mapk_p1_expproj: for (DATA_HW k_tile = 0; k_tile < tile_index[2]; k_tile++) {
                #pragma HLS LOOP_TRIPCOUNT min=7 max=28
                #pragma HLS PIPELINE II=1

                LOOP_mapl_p1_expproj: for (DATA_HW l_tile = 0; l_tile < HLS_tile_map; l_tile++) {
                    #pragma HLS UNROLL factor=4

                    if (l_tile < tile_index[2]) {
                        if (read_data[j_tile] == 0) {
                            valIn_data = ext_in_data.read();
                            temp = (ACT_CONV) (valIn_data.data >> quant[0]);
                            in_map[j_tile][k_tile][l_tile] = temp;

                            #pragma HLS RESOURCE variable=temp core=DSP48
                            temp = temp_map[i_tile][k_tile][l_tile] + convolution_1x1(temp, kernel_value);
                        } else {
                            temp = temp_map[i_tile][k_tile][l_tile] + convolution_1x1(in_map[j_tile][k_tile][l_tile], kernel_value);
                            #pragma HLS RESOURCE variable=temp core=DSP48
                        }

                        // Add bias and apply ReLU6
                        temp += bias_value;
                        if (info[0]) {
                            temp = ReLU6(temp, quant[2]);

                            valOut_data.data = (DATA_HW) temp;
                            valOut_data.keep = valIn_data.keep;
                            valOut_data.strb = valIn_data.strb;
                            valOut_data.user = valIn_data.user;
                            valOut_data.id = valIn_data.id;
                            valOut_data.dest = valIn_data.dest;
                            valOut_data.last = (info[16] && index + 1 == len) ? 1 : 0;
                            ext_out_data.write(valOut_data);
                        }
                    }
                }
            }

            // Manage residual map
            if (info[1]) {
                valIn_res = ext_residual_map_read.read();
                res_map_data = (ACT_CONV) valIn_res.data;
                temp += res_map_data;

                if (info[2]) {
                    valOut_res.data = (quant[3] >= 0) ? (DATA_HW) (temp << quant[3]) : (DATA_HW) (temp >> -quant[3]);
                    valOut_res.keep = valIn_res.keep;
                    valOut_res.strb = valIn_res.strb;
                    valOut_res.user = valIn_res.user;
                    valOut_res.id = valIn_res.id;
                    valOut_res.dest = valIn_res.dest;
                    valOut_res.last = (info[16] && index + 1 == len) ? 1 : 0;
                    ext_residual_map_write.write(valOut_res);
                }

                valOut_data.data = (DATA_HW) temp;
                valOut_data.keep = valIn_data.keep;
                valOut_data.strb = valIn_data.strb;
                valOut_data.user = valIn_data.user;
                valOut_data.id = valIn_data.id;
                valOut_data.dest = valIn_data.dest;
                valOut_data.last = (info[16] && index + 1 == len) ? 1 : 0;
                ext_out_data.write(valOut_data);
            }

            index++;
        }

        // Handle residual or compute missing data
        if (read_data[j_tile] == 0) {
            read_data[j_tile] = 1;
        }
    }

    // Ensure stream empties before exit
    if (!ext_out_data.empty()) {
        DATA_STREAM temp;
        do {
            #pragma HLS pipeline II=1
            temp = ext_out_data.read();
        } while (!temp.last);  // Read until last = 1
    }
}

void layer_depthwise(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
                     volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                     DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer) {

    // Tối ưu bộ nhớ
    ACT_CONV line_buffer[2][in_map_LEN];
    ACT_CONV window[3][3], reg[3];
    bool stride_x = (info[3] == 1), stride_y = (info[3] == 1);  // strides
    W_CONV kernel[9];   // kernel
    ACT_CONV temp;   // conv
    ACT_CONV bias_value;   // bias
    DATA_STREAM valIn_data, valOut_data;
    DATA_HW index = 0, len = info[0] * info[0] * tile_index[0], limit = info[1] * info[1];

    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0
    #pragma HLS ARRAY_PARTITION variable=reg complete
    #pragma HLS ARRAY_PARTITION variable=kernel complete
    #pragma HLS STREAM variable=ext_in_data depth=64
    #pragma HLS STREAM variable=ext_out_data depth=64

    LOOP_depth_depthwise: for (DATA_HW i_tile = 0, pos_kernel_base = 0; i_tile < tile_index[0]; i_tile++, pos_kernel_base += 9) {
        #pragma HLS LOOP_TRIPCOUNT min=4 max=32
        #pragma HLS PIPELINE II=1

        // Define kernel
        LOOP_kernel_depthwise: for (DATA_HW k = 0, pos_kernel = pos_kernel_base; k < 9; k++, pos_kernel++) {
            #pragma HLS LOOP_TRIPCOUNT min=9 max=9
            #pragma HLS UNROLL factor=4
            kernel[k] = ((EXT_DATA) ext_w_conv[pos_kernel + info[15]]).range(lower_index_CONV, 0);
            if (k == 0) {
                bias_value = (ACT_CONV) ((EXT_DATA) ext_b_conv[i_tile + info[14]]).range(upper_act_CONV, 0);
            }
        }

        // Read first values into line buffer
        DATA_HW read_count = info[1] + 2;
        LOOP_line_buffer_0_depthwise: for (DATA_HW k_tile = info[1] - 2; k_tile < info[1]; k_tile++) {
            #pragma HLS LOOP_TRIPCOUNT min=9 max=30
            #pragma HLS UNROLL factor=4
            valIn_data = ext_in_data.read();
            line_buffer[0][k_tile] = (ACT_CONV) (valIn_data.data >> quant[0]);
        }
        LOOP_line_buffer_1_depthwise: for (DATA_HW j_tile = 1; j_tile < 2; j_tile++) {
            #pragma HLS PIPELINE II=1
            LOOP_line_buffer_2_depthwise: for (DATA_HW k_tile = 0; k_tile < info[1]; k_tile++) {
                #pragma HLS LOOP_TRIPCOUNT min=9 max=30
                #pragma HLS UNROLL factor=4
                valIn_data = ext_in_data.read();
                line_buffer[j_tile][k_tile] = (ACT_CONV) (valIn_data.data >> quant[0]);
            }
        }

        // Copy values into window
        LOOP_window_x_depthwise: for (DATA_HW x = 1; x < 3; x++) {
            #pragma HLS PIPELINE II=1
            LOOP_window_y_depthwise: for (DATA_HW y = 1; y < 3; y++) {
                #pragma HLS UNROLL factor=4
                window[x][y] = line_buffer[x - 1][y + info[1] - 3];
            }
        }

        // Process map features
        LOOP_maps_depthwise: for (DATA_HW j_tile = 0, k_tile = 0; j_tile < info[1]; k_tile++) {
            #pragma HLS LOOP_TRIPCOUNT min=81 max=900
            #pragma HLS PIPELINE II=2
            if (j_tile > 0 && j_tile < info[10] && k_tile > 0 && k_tile < info[10] && stride_x && stride_y) {
                // convolution
                temp = 0;
                LOOP_conv_x_depthwise: for (DATA_HW x = 0, pos_x = j_tile - 1; x < 3; x++, pos_x++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LATENCY min=1 max=3

                    ACT_CONV partial_sum[3];  // Temporary registers
                    #pragma HLS ARRAY_PARTITION variable=partial_sum complete

                    if ((pos_x > 0 || info[6] == 0) && (pos_x < info[10] || info[7] == 0)) {
                        LOOP_conv_y_depthwise: for (DATA_HW y = 0; y < 3; y++) {
                            #pragma HLS PIPELINE II=1
                            partial_sum[y] = window[x][y] * kernel[y + x * 3];
                            #pragma HLS RESOURCE variable=partial_sum core=DSP48
                        }

                        // Tree reduction to improve timing
                        temp += (partial_sum[0] + partial_sum[1]) + partial_sum[2];
                    }
                }
                temp += bias_value;

                // ReLU6
                temp = ReLU6(temp, quant[2]);

                // Write out_data
                valOut_data.data = (DATA_HW) temp;
                valOut_data.keep = valIn_data.keep;
                valOut_data.strb = valIn_data.strb;
                valOut_data.user = valIn_data.user;
                valOut_data.id = valIn_data.id;
                valOut_data.dest = valIn_data.dest;
                valOut_data.last = (info[16] && index + 1 == len) ? 1 : 0;
                ext_out_data.write(valOut_data);
                index++;
            }

            // Shift line buffer
            reg[0] = line_buffer[0][k_tile];
            reg[1] = line_buffer[0][k_tile] = line_buffer[1][k_tile];

            // Read input data
            if (read_count < limit) {
                valIn_data = ext_in_data.read();
                read_count++;
            }

            reg[2] = line_buffer[1][k_tile] = (ACT_CONV) (valIn_data.data >> quant[0]);

            // Shift window
            LOOP_shift_window_x_depthwise: for (DATA_HW x = 0; x < 3; x++) {
                #pragma HLS PIPELINE II=1
                LOOP_shift_window_y_depthwise: for (DATA_HW y = 0; y < 2; y++) {
                    window[x][y] = window[x][y + 1];
                }
                window[x][2] = reg[x];
            }

            // Stride x
            if (info[3] == 2) {
                stride_x = !stride_x;
            }

            // New index
            if (k_tile + 1 == info[1]) {
                k_tile = -1;
                j_tile++;

                // Stride y
                if (info[3] == 2) {
                    stride_y = !stride_y;
                }
            }
        }
    }

    // Ensure stream empties before exit
    if (!ext_out_data.empty()) {
        DATA_STREAM temp;
        do {
            #pragma HLS pipeline II=1
            temp = ext_out_data.read();
        } while (!temp.last);  // Read until last = 1
    }
}


void layer_AVG(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
               DATA_HW tile_index[3], DATA_HW info[size_info], DATA_HW quant[3], DATA_HW type_layer) {

    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=2
    #pragma HLS LATENCY min=1 max=3
    #pragma HLS STREAM variable=ext_in_data depth=64
    #pragma HLS STREAM variable=ext_out_data depth=64

    static ACT_FC temp = 0;
    static DATA_HW i_tile = 0, j_tile = 0, k_tile = 0;
    static bool last_tile = false;

    // Precompute constants and boundaries
    const DATA_HW k_limit = tile_index[1];
    const DATA_HW j_limit = tile_index[1];
    const DATA_HW i_limit = tile_index[0];
    const int quant_shift = quant[0];
    const bool final_layer = info[9];

    DATA_STREAM valIn_data, valOut_data;

    // Read input data
    valIn_data = ext_in_data.read();
    ACT_CONV in_map_data = (ACT_CONV)(valIn_data.data >> quant_shift);

    // Accumulate with registered output to break critical path
    ACT_FC temp_next = temp + in_map_data;
    temp = temp_next;

    // Update counters with registered outputs
    DATA_HW k_tile_next = k_tile + 1;
    bool k_tile_done = (k_tile_next == k_limit);

    DATA_HW j_tile_next = k_tile_done ? j_tile + 1 : j_tile;
    bool j_tile_done = (j_tile_next == j_limit);

    DATA_HW i_tile_next = (k_tile_done && j_tile_done) ? i_tile + 1 : i_tile;
    last_tile = (final_layer && (i_tile_next == i_limit));

    // Update registers
    k_tile = k_tile_done ? 0 : k_tile_next;
    j_tile = k_tile_done ? (j_tile_done ? 0 : j_tile_next) : j_tile;
    i_tile = (k_tile_done && j_tile_done) ? i_tile_next : i_tile;

    // Output result when completing a tile
    if (k_tile_done && j_tile_done) {
        // Use right shift approximation for division by 49 (2^6 = 64 is close)
        valOut_data.data = (DATA_SW)((temp + 24) >> 6); // Rounding by adding 24 before shift
        valOut_data.keep = valIn_data.keep;
        valOut_data.strb = valIn_data.strb;
        valOut_data.user = valIn_data.user;
        valOut_data.id = valIn_data.id;
        valOut_data.dest = valIn_data.dest;
        valOut_data.last = last_tile;

        ext_out_data.write(valOut_data);
        temp = 0;
    }

    // Ensure stream empties before exit
    if (!ext_out_data.empty()) {
        DATA_STREAM temp;
        do {
            #pragma HLS pipeline II=1
            temp = ext_out_data.read();
        } while (!temp.last);  // Read until last = 1
    }
}


void layer_FC(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
              volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
              DATA_HW tile_index[3], DATA_HW quant[4], DATA_HW info[size_info], DATA_HW type_layer) {
    #pragma HLS STREAM variable=ext_in_data depth=64
    #pragma HLS STREAM variable=ext_out_data depth=64

    static ACT_FC temp_array[tile_fc_out];
    ACT_FC in_array[tile_fc_in];
    ACT_FC temp;   // FC
    DATA_STREAM valIn_data, valOut_data;

    // weights
    W_FC w_fc[w_fc_LEN];
    ACT_FC b_fc[tile_fc_out];
    I_FC i_fc[w_fc_LEN];

    #pragma HLS RESOURCE variable=w_fc core=XPM_MEMORY uram
    #pragma HLS RESOURCE variable=b_fc core=XPM_MEMORY uram
    #pragma HLS RESOURCE variable=i_fc core=XPM_MEMORY uram

    // read weights
    read_w_fc(ext_w_fc, ext_b_fc, w_fc, i_fc, b_fc, tile_index[0] * info[3], tile_index[0], info, type_layer);

    LOOP_fc: for (DATA_HW i_tile = 0, j_tile = 0, distance = 0, init_signal = 1, pos_weight = 0; i_tile < tile_index[0]; j_tile++) {
        #pragma HLS LOOP_TRIPCOUNT min=32*64 max=32*64
        #pragma HLS PIPELINE II=1

        // read in_data
        if (i_tile == 0) {
            valIn_data = ext_in_data.read();
            in_array[j_tile] = (ACT_FC) (valIn_data.data >> quant[0]);
        }

        // Process
        if (distance == i_fc[pos_weight]) {
            if (info[2] && init_signal) {
                temp = (in_array[j_tile]) * w_fc[pos_weight];
                #pragma HLS RESOURCE variable=temp core=DSP48
                init_signal = 0;
            } else if (info[2] == 0 && init_signal) {
                temp = temp_array[i_tile] + (in_array[j_tile]) * w_fc[pos_weight];
                init_signal = 0;
            } else {
                temp += (in_array[j_tile]) * w_fc[pos_weight];
            }
            pos_weight++;
            distance = 0;
        } else {
            distance++;
        }

        // New index
        if (j_tile + 1 == tile_index[1]) {
            j_tile = -1;
            distance = 0;
            init_signal = 1;

            // bias
            if (info[1] >= info[0]) {
                // write out_data
                valOut_data.data = (DATA_SW) temp + b_fc[i_tile];
                valOut_data.keep = valIn_data.keep;
                valOut_data.strb = valIn_data.strb;
                valOut_data.user = valIn_data.user;
                valOut_data.id = valIn_data.id;
                valOut_data.dest = valIn_data.dest;
                valOut_data.last = (info[9] && i_tile + 1 == tile_index[0]) ? 1 : 0;
                ext_out_data.write(valOut_data);
            } else {
                temp_array[i_tile] = temp;
            }
            i_tile++;
        }
    }

    // Ensure stream empties before exit
    if (!ext_out_data.empty()) {
        DATA_STREAM temp;
        do {
            #pragma HLS pipeline II=1
            temp = ext_out_data.read();
        } while (!temp.last);  // Read until last = 1
    }
}



//MATH

ACT_CONV convolution_1x1(ACT_CONV in_map, W_CONV kernel) {
    #pragma HLS INLINE
    #pragma HLS RESOURCE variable=return core=DMul_meddsp latency=2
    return in_map * kernel;
}


// Đảm bảo tất cả hàm được gán DSP đều có return value

ACT_CONV ReLU6(ACT_CONV in_map, DATA_HW upper) {
    #pragma HLS INLINE
    #pragma HLS EXPRESSION_BALANCE
    if (in_map <= 0) {
        return 0;
    } else if (in_map >= upper) {
        return upper;
    } else {
        return in_map;
    }
}


DATA_HW MIN(DATA_HW x, DATA_HW y){
	#pragma HLS INLINE

    if (x < y){
        return x;
    }
    else{
        return y;
    }
}


//
//Read AXI master
//

void read_w_conv(volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
		W_CONV w_conv[w_conv_LEN], I_CONV i_conv[w_conv_LEN], ACT_CONV b_conv[tile_conv_out],
		DATA_HW tile_index[3], DATA_HW info[size_info]){

	if (info[18]){
		DATA_HW temp;
		DATA_HW LEN_W = tile_index[0]*info[3];
		DATA_HW LEN_B = tile_index[0];
		//read W_CONV
        #pragma HLS INTERFACE m_axi port=ext_w_conv latency=32 num_read_outstanding=16
        #pragma HLS LOOP_FLATTEN
 		for (DATA_HW i = 0; i < LEN_W; i++){
			#pragma HLS LOOP_TRIPCOUNT min=96 max=1024
			#pragma HLS PIPELINE II=1
			temp = ext_w_conv[i + info[15]];
			w_conv[i] = ((EXT_DATA) temp).range(lower_index_CONV, 0);
			i_conv[i] = ((EXT_DATA) temp).range(upper_index_CONV, bit_w_CONV);
			if (i < LEN_B && info[17]){
				b_conv[i] = (ACT_CONV) ((EXT_DATA) ext_b_conv[i + info[14]]).range(upper_act_CONV, 0);
			}
		}
	}
}

void read_w_fc(volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
               W_FC w_fc[w_fc_LEN], I_FC i_fc[w_fc_LEN], ACT_FC b_fc[tile_fc_out],
               DATA_HW LEN_W, DATA_HW LEN_B, DATA_HW info[size_info], DATA_HW type_layer) {

    DATA_HW temp;

    for (DATA_HW i = 0; i < LEN_W; i++) {
        #pragma HLS LOOP_TRIPCOUNT max=32*64
        #pragma HLS PIPELINE II=1

        if (i + info[6] >= LEN_W) {
            continue;
        }

        temp = ext_w_fc[i + info[6]];
        w_fc[i] = ((EXT_DATA) temp).range(lower_index_FC, 0);
        i_fc[i] = ((EXT_DATA) temp).range(upper_index_FC, bit_w_FC);

        if (i < LEN_B) {
            b_fc[i] = (ACT_FC) ((EXT_DATA) ext_b_fc[i + info[5]]).range(upper_act_CONV, 0);
        }
    }
}




DATA_STREAM read_in_map(hls::stream<DATA_STREAM> &ext_in_data,
				 	 ACT_CONV in_map[tile_conv_out][in_map_LEN][in_map_LEN],
					 DATA_HW tile_index[3], DATA_HW info[size_info], DATA_HW quant, DATA_HW type_layer){

	DATA_STREAM valIn;
	DATA_HW len, limit;

	if (type_layer == 0){
		len = info[1]*info[1]*tile_index[1];
		limit = info[1];
	}
	else{
		len = info[1]*info[1]*tile_index[0];
		limit=info[1];
	}

	//read IN_MAP
	for (DATA_HW index = 0, i = 0, j = 0, k = 0; index <len; index++, k++){
		#pragma HLS LOOP_TRIPCOUNT min=3*HLS_tile_map*HLS_tile_map max=HLS_tile_conv_out*HLS_tile_map*HLS_tile_map
		#pragma HLS PIPELINE II=1
		if (k == limit){
			k = 0;
			j++;
			if (j == limit){
				j = 0;
				i++;
			}
		}
		valIn = ext_in_data.read();
		in_map[i][j][k] = (ACT_CONV) (valIn.data >> quant);
	}

	return valIn;
}

void generate_quant(DATA_SW layer, DATA_SW inter_layer, DATA_SW type_layer,
                    DATA_HW quant[4]) {

    if (type_layer < 3) {
        for (DATA_HW i = 0; i < 4; i++) {
            #pragma HLS PIPELINE II=1
            quant[i] = CONV_quant[layer][inter_layer][i];
        }
    } else if (type_layer == 3) {
        quant[0] = AVG_quant;
    } else {
        for (DATA_HW i = 0; i < 2; i++) {
            #pragma HLS PIPELINE II=1
            quant[i] = FC_quant[i];
        }
    }
}


void generate_info_tile(volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
                        DATA_HW tile_index[3], DATA_HW info[size_info],
                        CALL_DATA call_PE, DATA_HW type_layer) {

    if (call_PE < MAX_CALL[type_layer]) {
        generate_tile(ext_tile, tile_index, call_PE, type_layer);
        generate_info(ext_info, info, call_PE, type_layer);
    }
}


void generate_tile(volatile DATA_SW *ext_tile,
                   DATA_HW tile_index[3],
                   CALL_DATA call_PE, DATA_HW type_layer) {
    DATA_HW offset = call_PE * 3;

    for (DATA_HW i = 0; i < 3; i++) {
        #pragma HLS PIPELINE
        tile_index[i] = (DATA_HW) ext_tile[offset + i];
    }
}


void generate_info(volatile DATA_SW *ext_info,
                   DATA_HW info[size_info],
                   CALL_DATA call_PE, DATA_HW type_layer) {
    DATA_HW offset = call_PE * size_info;

    for (DATA_HW i = 0; i < size_info; i++) {
        #pragma HLS PIPELINE
        info[i] = (DATA_HW) ext_info[offset + i];
    }
}

//
//PE function
//

void PEs(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
         hls::stream<DATA_STREAM> &ext_residual_map_read, hls::stream<DATA_STREAM> &ext_residual_map_write,
         volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
         volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
         DATA_HW tile_index[3], DATA_SW quant[4], DATA_SW info[size_info], CALL_DATA call_PE, DATA_SW type_layer) {

    if (call_PE < MAX_CALL[type_layer]) {
        if (type_layer == 0) {
            layer_CONV_3x3(ext_in_data, ext_out_data, ext_w_conv, ext_b_conv, tile_index, quant, info, type_layer);
        } else if (type_layer == 1) {
            layer_expansion_projection(ext_in_data, ext_out_data, ext_residual_map_read, ext_residual_map_write,
                                       ext_w_conv, ext_b_conv, tile_index, quant, info, type_layer);
        } else if (type_layer == 2) {
            layer_depthwise(ext_in_data, ext_out_data, ext_w_conv, ext_b_conv, tile_index, quant, info, type_layer);
        } else if (type_layer == 3) {
            layer_AVG(ext_in_data, ext_out_data, tile_index, info, quant, type_layer);
        } else if (type_layer == 4) {
            layer_FC(ext_in_data, ext_out_data, ext_w_fc, ext_b_fc, tile_index, quant, info, type_layer);
        }
    }
}

void MobileNet_Stream(hls::stream<DATA_STREAM> &ext_in_data, hls::stream<DATA_STREAM> &ext_out_data,
                      hls::stream<DATA_STREAM> &ext_residual_map_write, hls::stream<DATA_STREAM> &ext_residual_map_read,
                      volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                      volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
                      volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
                      DATA_HW layer, DATA_SW inter_layer, DATA_SW type_layer) {
    if (ext_in_data.empty()) {
        DATA_STREAM dummy;
        dummy.data = 0;
        dummy.last = 1;
        ext_out_data.write(dummy);
        return;
    }

    #pragma HLS INTERFACE s_axilite port=layer bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=inter_layer bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=type_layer bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
    #pragma HLS INTERFACE axis port=ext_in_data bundle=IN_MAP
    #pragma HLS INTERFACE axis port=ext_out_data bundle=OUT_MAP
    #pragma HLS INTERFACE axis port=ext_residual_map_read bundle=RESIDUAL_MAP_READ
    #pragma HLS INTERFACE axis port=ext_residual_map_write bundle=RESIDUAL_MAP_WRITE
    #pragma HLS INTERFACE m_axi port=ext_w_conv depth=w_conv_layer offset=slave bundle=W_CONV
    #pragma HLS INTERFACE m_axi port=ext_b_conv depth=b_conv_layer offset=slave bundle=B_CONV
    #pragma HLS INTERFACE m_axi port=ext_w_fc depth=fc_layer offset=slave bundle=W_FC
    #pragma HLS INTERFACE m_axi port=ext_b_fc depth=b_fc_layer offset=slave bundle=B_FC
    #pragma HLS INTERFACE m_axi port=ext_tile depth=HLS_tile_size offset=slave bundle=TILE
    #pragma HLS INTERFACE m_axi port=ext_info depth=HLS_info_size offset=slave bundle=INFO
    #pragma HLS DATAFLOW
    #pragma HLS RESOURCE variable=ext_in_data core=FIFO_LUTRAM

    DATA_HW quant[4];
    #pragma HLS ARRAY_PARTITION variable=quant complete

    // Generate info and tile for 4 PEs
    DATA_HW info_0[size_info], tile_index_0[3];
    #pragma HLS ARRAY_PARTITION variable=info_0 complete
    #pragma HLS ARRAY_PARTITION variable=tile_index_0 complete

    DATA_HW info_1[size_info], tile_index_1[3];
    #pragma HLS ARRAY_PARTITION variable=info_1 complete
    #pragma HLS ARRAY_PARTITION variable=tile_index_1 complete

    DATA_HW info_2[size_info], tile_index_2[3];
    #pragma HLS ARRAY_PARTITION variable=info_2 complete
    #pragma HLS ARRAY_PARTITION variable=tile_index_2 complete

    DATA_HW info_3[size_info], tile_index_3[3];
    #pragma HLS ARRAY_PARTITION variable=info_3 complete
    #pragma HLS ARRAY_PARTITION variable=tile_index_3 complete

    for (CALL_DATA i = 0; i < MAX_FC; i += stage_pipeline * 2) {
        generate_info_tile(ext_tile, ext_info, tile_index_0, info_0, i, type_layer);
        generate_info_tile(ext_tile, ext_info, tile_index_1, info_1, i + 1, type_layer);
        generate_info_tile(ext_tile, ext_info, tile_index_2, info_2, i + 2, type_layer);
        generate_info_tile(ext_tile, ext_info, tile_index_3, info_3, i + 3, type_layer);

        PEs(ext_in_data, ext_out_data, ext_residual_map_read, ext_residual_map_write,
            ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc,
            tile_index_0, quant, info_0, i, type_layer);
        PEs(ext_in_data, ext_out_data, ext_residual_map_read, ext_residual_map_write,
            ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc,
            tile_index_1, quant, info_1, i + 1, type_layer);
        PEs(ext_in_data, ext_out_data, ext_residual_map_read, ext_residual_map_write,
            ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc,
            tile_index_2, quant, info_2, i + 2, type_layer);
        PEs(ext_in_data, ext_out_data, ext_residual_map_read, ext_residual_map_write,
            ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc,
            tile_index_3, quant, info_3, i + 3, type_layer);
    }

    if (ext_out_data.empty()) {
        DATA_STREAM term_pkt;
        term_pkt.data = 0;
        term_pkt.last = 1;
        ext_out_data.write(term_pkt);
    }
}
