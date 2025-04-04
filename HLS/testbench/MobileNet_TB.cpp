#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <hls_stream.h>
#include <cmath>
#include <chrono>
#include "D:/Tailieu/DA2/MobilenetV2_HLS/parameters.h"

using namespace std;

// Khai báo hàm kernel chính
void MobileNet_Stream(hls::stream<DATA_STREAM> &ext_in_data,
                      hls::stream<DATA_STREAM> &ext_out_data,
                      hls::stream<DATA_STREAM> &ext_residual_map_write,
                      hls::stream<DATA_STREAM> &ext_residual_map_read,
                      volatile DATA_SW *ext_w_conv, volatile DATA_SW *ext_b_conv,
                      volatile DATA_SW *ext_w_fc, volatile DATA_SW *ext_b_fc,
                      volatile DATA_SW *ext_tile, volatile DATA_SW *ext_info,
                      DATA_HW layer, DATA_SW inter_layer, DATA_SW type_layer);

// === Tiện ích ===
vector<int> load_image(const string &filename) {
    ifstream file(filename);
    vector<int> image_data;
    int val;
    while (file >> val) image_data.push_back(val);
    return image_data;
}

vector<string> load_labels(const string &filename) {
    ifstream file(filename);
    vector<string> labels;
    string line;
    while (getline(file, line)) labels.push_back(line);
    return labels;
}

vector<int> load_int_vector(const string &filename) {
    ifstream file(filename);
    vector<int> vec;
    int val;
    while (file >> val) vec.push_back(val);
    return vec;
}

int argmax(const vector<int> &vec) {
    int max_val = vec[0], max_idx = 0;
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] > max_val) {
            max_val = vec[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// === MAIN ===
int main() {
    const int num_images = 6;
    const string image_files[num_images] = {
        "image_int1.dat", "image_int2.dat", "image_int3.dat",
        "image_int4.dat", "image_int5.dat", "image_int6.dat"
    };

    vector<string> labels = load_labels("imagenet_class.txt");
    vector<int> weights = load_int_vector("weights_fixed.dat");
    vector<int> biases  = load_int_vector("bias_fixed.dat");
    vector<int> tiles   = load_int_vector("tile.dat");
    vector<int> infos   = load_int_vector("info.dat");

    static DATA_SW ext_w_conv[w_conv_layer] = {0};
    static DATA_SW ext_b_conv[b_conv_layer] = {0};
    static DATA_SW ext_w_fc[fc_layer] = {0};
    static DATA_SW ext_b_fc[b_fc_layer] = {0};
    static DATA_SW ext_tile[HLS_tile_size] = {0};
    static DATA_SW ext_info[HLS_info_size] = {0};

    // Load weight/bias
    for (int i = 0; i < min((int)weights.size(), w_conv_layer); ++i)
        ext_w_conv[i] = weights[i];
    for (int i = 0; i < min((int)biases.size(), b_conv_layer); ++i)
        ext_b_conv[i] = biases[i];

    // Load tile & info
    for (int i = 0; i < min((int)tiles.size(), HLS_tile_size); ++i)
        ext_tile[i] = tiles[i];
    for (int i = 0; i < min((int)infos.size(), HLS_info_size); ++i)
        ext_info[i] = infos[i];

    // FC mặc định
    for (int i = 0; i < fc_layer; ++i) ext_w_fc[i] = 1;
    for (int i = 0; i < b_fc_layer; ++i) ext_b_fc[i] = 0;

    int correct_predictions = 0;
    double total_latency_ms = 0.0;

    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        cout << "\n==== Processing Image " << (img_idx + 1) << " ====" << endl;
        vector<int> img_data = load_image(image_files[img_idx]);

        hls::stream<DATA_STREAM> stream_input;
        hls::stream<DATA_STREAM> stream_output;
        hls::stream<DATA_STREAM> residual_read;
        hls::stream<DATA_STREAM> residual_write;

        // Load ảnh vào stream_input
        for (size_t i = 0; i < img_data.size(); ++i) {
            DATA_STREAM pkt;
            pkt.data = img_data[i];
            pkt.last = (i == img_data.size() - 1) ? 1 : 0;
            stream_input.write(pkt);
        }

        auto start = chrono::high_resolution_clock::now();

        // === Chuỗi pipeline MobileNet ===
        hls::stream<DATA_STREAM> conv_output;
        hls::stream<DATA_STREAM> exp_output;
        hls::stream<DATA_STREAM> dw_output;
        hls::stream<DATA_STREAM> avg_output;
        hls::stream<DATA_STREAM> fc_output;

        // Layer 0: CONV_3x3
        MobileNet_Stream(stream_input, conv_output, residual_write, residual_read,
                         ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc,
                         ext_tile, ext_info,
                         0, 0, 0);

        // Chuyển dữ liệu conv_output → exp_input
        hls::stream<DATA_STREAM> exp_input;
        while (!conv_output.empty()) exp_input.write(conv_output.read());

        // Layer 1: Expansion/Projection
        MobileNet_Stream(exp_input, exp_output, residual_write, residual_read,
                         ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc,
                         ext_tile, ext_info,
                         1, 0, 1);

        // Chuyển dữ liệu exp_output → dw_input
        hls::stream<DATA_STREAM> dw_input;
        while (!exp_output.empty()) dw_input.write(exp_output.read());

        // Layer 2: Depthwise
        MobileNet_Stream(dw_input, dw_output, residual_write, residual_read,
                         ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc,
                         ext_tile, ext_info,
                         2, 0, 2);

        // Layer 3: AVG Pooling
        hls::stream<DATA_STREAM> avg_input;
        while (!dw_output.empty()) avg_input.write(dw_output.read());

        MobileNet_Stream(avg_input, avg_output, residual_write, residual_read,
                         ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc,
                         ext_tile, ext_info,
                         3, 0, 3);

        // Layer 4: FC
        hls::stream<DATA_STREAM> fc_input;
        while (!avg_output.empty()) fc_input.write(avg_output.read());

        MobileNet_Stream(fc_input, fc_output, residual_write, residual_read,
                         ext_w_conv, ext_b_conv, ext_w_fc, ext_b_fc,
                         ext_tile, ext_info,
                         4, 0, 4);

        auto end = chrono::high_resolution_clock::now();
        double latency_ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
        total_latency_ms += latency_ms;

        // === Đọc output cuối cùng ===
        vector<int> output_scores;
        while (!fc_output.empty()) {
            DATA_STREAM d = fc_output.read();
            output_scores.push_back(d.data);
        }

        if (output_scores.empty()) {
            cerr << "Lỗi: không có dữ liệu output!" << endl;
            continue;
        }

        int predicted = argmax(output_scores);
        cout << "→ Predicted: " << labels[predicted] << " (class " << predicted << "), Latency: " << latency_ms << " ms\n";

        // Đánh giá (giả sử ground truth = img_idx)
        if (predicted == img_idx) correct_predictions++;
    }

    // === Tổng kết ===
    double avg_latency = total_latency_ms / num_images;
    double fps = 1000.0 / avg_latency;
    double accuracy = 100.0 * correct_predictions / num_images;

    cout << "\n=== Evaluation Results ===" << endl;
    cout << "Average Latency  : " << avg_latency << " ms" << endl;
    cout << "FPS              : " << fps << endl;
    cout << "Accuracy         : " << accuracy << " %" << endl;

    return 0;
}
