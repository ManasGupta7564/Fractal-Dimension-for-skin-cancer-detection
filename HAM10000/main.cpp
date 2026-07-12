
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "fractal.hpp"

// Compile: g++ main.cpp fractal.cpp -o fractal_app `pkg-config --cflags --libs opencv4`

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./fractal_app <image_path> <output_csv>\n";
        return 1;
    }

    std::string image_path = argv[1];
    std::string output_csv = argv[2];

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }

    if (img.cols < M || img.rows < M) {
        std::cerr << "Image too small for sliding window (" << M << "x" << M << ")\n";
        return 1;
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    int step = 4;  // stride for window
    int out_rows = (img.rows - M) / step + 1;
    int out_cols = (img.cols - M) / step + 1;

    std::vector<std::vector<double>> fractal_map(out_rows, std::vector<double>(out_cols, 0.0));

    for (int y = 0; y <= img.rows - M; y += step) {
        for (int x = 0; x <= img.cols - M; x += step) {
            std::array<std::array<std::array<unsigned char, M>, M>, 3> window{};

            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < M; ++j) {
                    cv::Vec3b pixel = img.at<cv::Vec3b>(y + i, x + j);
                    window[0][i][j] = pixel[0]; // R
                    window[1][i][j] = pixel[1]; // G
                    window[2][i][j] = pixel[2]; // B
                }
            }

            int row_idx = y / step;
            int col_idx = x / step;
            fractal_map[row_idx][col_idx] = get_fractal_dimension(window);
        }
    }

    std::ofstream file(output_csv);
    for (const auto &row : fractal_map) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j != row.size() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();

    std::cout << "Saved: " << output_csv << "\n";
    return 0;
}
