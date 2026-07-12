#ifndef FRACTAL_HPP
#define FRACTAL_HPP

#include <array>
#include <vector>

constexpr int M = 32;   // image size 512x512
constexpr int G = 256;  // gray levels (0-255)

// Function declarations
void DBC_RGB(const std::array<std::array<std::array<unsigned char, M>, M>, 3>& I,
             std::vector<unsigned long>& Nr,
             std::vector<int>& box_sizes,
             int& num_scales);

double compute_fractal_dimension(const std::vector<unsigned long>& Nr,
                                 const std::vector<int>& box_sizes,
                                 int num_scales, int img_size);

double get_fractal_dimension(const std::array<std::array<std::array<unsigned char, M>, M>, 3>& I);

#endif // FRACTAL_HPP
