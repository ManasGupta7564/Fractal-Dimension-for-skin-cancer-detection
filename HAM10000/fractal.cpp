#include "fractal.hpp"
#include <algorithm>
#include <cmath>

void DBC_RGB(const std::array<std::array<std::array<unsigned char, M>, M>, 3>& I,
             std::vector<unsigned long>& Nr,
             std::vector<int>& box_sizes,
             int& num_scales) 
{
    std::array<std::array<unsigned char, M>, M> ImaxR, IminR;
    std::array<std::array<unsigned char, M>, M> ImaxG, IminG;
    std::array<std::array<unsigned char, M>, M> ImaxB, IminB;

    // Copy RGB channels
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            ImaxR[i][j] = IminR[i][j] = I[0][i][j];
            ImaxG[i][j] = IminG[i][j] = I[1][i][j];
            ImaxB[i][j] = IminB[i][j] = I[2][i][j];
        }
    }

    unsigned int s = 2;
    unsigned int size = M;
    unsigned int Nri = 0;

    while (size > 2) {
        int h = (G * s) / M;

        for (unsigned long i = 0; i < (M - 1); i += s) {
            for (unsigned long j = 0; j < (M - 1); j += s) {
                ImaxR[i][j] = std::max({ImaxR[i][j], ImaxR[i + s/2][j],
                                        ImaxR[i][j + s/2], ImaxR[i + s/2][j + s/2]});
                IminR[i][j] = std::min({IminR[i][j], IminR[i + s/2][j],
                                        IminR[i][j + s/2], IminR[i + s/2][j + s/2]});

                ImaxG[i][j] = std::max({ImaxG[i][j], ImaxG[i + s/2][j],
                                        ImaxG[i][j + s/2], ImaxG[i + s/2][j + s/2]});
                IminG[i][j] = std::min({IminG[i][j], IminG[i + s/2][j],
                                        IminG[i][j + s/2], IminG[i + s/2][j + s/2]});

                ImaxB[i][j] = std::max({ImaxB[i][j], ImaxB[i + s/2][j],
                                        ImaxB[i][j + s/2], ImaxB[i + s/2][j + s/2]});
                IminB[i][j] = std::min({IminB[i][j], IminB[i + s/2][j],
                                        IminB[i][j + s/2], IminB[i + s/2][j + s/2]});

                Nr[Nri] += (ImaxR[i][j] / h - IminR[i][j] / h + 1) *
                           (ImaxG[i][j] / h - IminG[i][j] / h + 1) *
                           (ImaxB[i][j] / h - IminB[i][j] / h + 1);
            }
        }

        box_sizes[Nri] = s;
        Nri++;
        s *= 2;
        size = M / s;
    }

    num_scales = Nri;
}

double compute_fractal_dimension(const std::vector<unsigned long>& Nr,
                                 const std::vector<int>& box_sizes,
                                 int num_scales, int img_size) 
{
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

    for (int i = 0; i < num_scales; i++) {
        double inv_r = static_cast<double>(box_sizes[i]) / img_size;
        double log_inv_r = std::log(1.0 / inv_r);
        double log_Nr = std::log(static_cast<double>(Nr[i]));

        sum_x += log_inv_r;
        sum_y += log_Nr;
        sum_xy += log_inv_r * log_Nr;
        sum_x2 += log_inv_r * log_inv_r;
    }

    return (num_scales * sum_xy - sum_x * sum_y) /
           (num_scales * sum_x2 - sum_x * sum_x);
}

double get_fractal_dimension(const std::array<std::array<std::array<unsigned char, M>, M>, 3>& I) {
    std::vector<unsigned long> Nr(20, 0);
    std::vector<int> box_sizes(20, 0);
    int num_scales = 0;

    DBC_RGB(I, Nr, box_sizes, num_scales);
    return compute_fractal_dimension(Nr, box_sizes, num_scales, M);
}
