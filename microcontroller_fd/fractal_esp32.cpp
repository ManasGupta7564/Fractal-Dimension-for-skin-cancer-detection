#include "fractal_esp32.h"
#include <math.h>

// Helper macros (faster than std::max/min)
#define MAX4(a,b,c,d) ( (a>b ? (a>c ? (a>d?a:d) : (c>d?c:d)) : (b>c ? (b>d?b:d) : (c>d?c:d))) )
#define MIN4(a,b,c,d) ( (a<b ? (a<c ? (a<d?a:d) : (c<d?c:d)) : (b<c ? (b<d?b:d) : (c<d?c:d))) )

void DBC_RGB(ImageRGB img,
             unsigned long Nr[MAX_SCALES],
             int box_sizes[MAX_SCALES],
             int *num_scales)
{
    int s = 2;
    int Nri = 0;

    while (s <= M) {
        int h = (G * s) / M;

        unsigned long count = 0;

        for (int i = 0; i < M; i += s) {
            for (int j = 0; j < M; j += s) {

                int i2 = i + s/2;
                int j2 = j + s/2;

                // Bound safety (important for MCU)
                if (i2 >= M) i2 = M - 1;
                if (j2 >= M) j2 = M - 1;

                // ---- R channel ----
                uint8_t r1 = img[0][i][j];
                uint8_t r2 = img[0][i2][j];
                uint8_t r3 = img[0][i][j2];
                uint8_t r4 = img[0][i2][j2];

                uint8_t ImaxR = MAX4(r1, r2, r3, r4);
                uint8_t IminR = MIN4(r1, r2, r3, r4);

                // ---- G channel ----
                uint8_t g1 = img[1][i][j];
                uint8_t g2 = img[1][i2][j];
                uint8_t g3 = img[1][i][j2];
                uint8_t g4 = img[1][i2][j2];

                uint8_t ImaxG = MAX4(g1, g2, g3, g4);
                uint8_t IminG = MIN4(g1, g2, g3, g4);

                // ---- B channel ----
                uint8_t b1 = img[2][i][j];
                uint8_t b2 = img[2][i2][j];
                uint8_t b3 = img[2][i][j2];
                uint8_t b4 = img[2][i2][j2];

                uint8_t ImaxB = MAX4(b1, b2, b3, b4);
                uint8_t IminB = MIN4(b1, b2, b3, b4);

                // DBC contribution
                int NrR = (ImaxR / h - IminR / h + 1);
                int NrG = (ImaxG / h - IminG / h + 1);
                int NrB = (ImaxB / h - IminB / h + 1);

                count += (unsigned long)(NrR * NrG * NrB);
            }
        }

        Nr[Nri] = count;
        box_sizes[Nri] = s;

        Nri++;
        s *= 2;

        if (Nri >= MAX_SCALES) break;
    }

    *num_scales = Nri;
}
static const float log_inv_r_table[4] = {
    2.07944f,   // log(8)
    1.38629f,   // log(4)
    0.69315f,   // log(2)
    0.0f        // log(1)
};

// ---------------- FD COMPUTATION ----------------

float compute_fractal_dimension(unsigned long Nr[MAX_SCALES],
                                int box_sizes[MAX_SCALES],
                                int num_scales,
                                int img_size)
{
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;

    for (int i = 0; i < num_scales; i++) {
        // float r = (float)box_sizes[i] / img_size;
        // float log_inv_r = logf(1.0f / r);

        float log_inv_r = log_inv_r_table[i];


        float log_Nr = logf((float)Nr[i]);

        sum_x += log_inv_r;
        sum_y += log_Nr;
        sum_xy += log_inv_r * log_Nr;
        sum_x2 += log_inv_r * log_inv_r;
    }

    float numerator = num_scales * sum_xy - sum_x * sum_y;
    float denominator = num_scales * sum_x2 - sum_x * sum_x;

    return numerator / denominator;
}


// ---------------- WRAPPER ----------------

float get_fractal_dimension(ImageRGB img)
{
    unsigned long Nr[MAX_SCALES] = {0};
    int box_sizes[MAX_SCALES] = {0};
    int num_scales = 0;

    DBC_RGB(img, Nr, box_sizes, &num_scales);

    return compute_fractal_dimension(Nr, box_sizes, num_scales, M);
}