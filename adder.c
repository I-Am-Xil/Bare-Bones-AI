#define NN_IMPLEMENTATION

#include<stdio.h>
#include"libraries/nn/nn.h"
#include<time.h>

#define BITS 4

float td[] = {
    1
};

int main() {
    srand(time(0));

    size_t n = (1<<BITS);
    size_t rows = n*n;
    Mat ti = mat_alloc(rows, 2*BITS);
    Mat to = mat_alloc(rows, BITS + 1);

    for (size_t i = 0; i < ti.rows; i++) {
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;
        size_t overflow = z >= n;
        for (size_t j = 0; j < BITS; j++) {
            MAT_AT(ti, i, j) = (x>>j)&1;
            MAT_AT(ti, i, j + BITS) = (y>>j)&1;
            if (overflow) {
                MAT_AT(to, i, j) = 0;
            } else {
                MAT_AT(to, i, j) = (z >>j)&1;
            }
        }
        MAT_AT(to, i, BITS) = overflow;
    }
    
    size_t arch[] = {2*BITS, 2*BITS, BITS + 1};
    NN nn;
    NN g;

    (void) nn_alloc(&nn, arch, ARRAY_LEN(arch));
    (void) nn_alloc(&g, arch, ARRAY_LEN(arch));

    nn_rand(&nn, 0, 1);

    NN_PRINT(nn);

    float rate = 1;

    for (size_t i = 0; i < 10000; i++) {
        nn_backprop(&nn, &g, ti, to);
        //nn_finite_diff(&nn, &g, 1e-1, ti, to);
        nn_learn(&nn, &g, rate);
        //printf("%u c = %f\n", i, nn_cost(&nn, ti, to));
    }
    printf("c = %f\n", nn_cost(&nn, ti, to));
    size_t fails = 0;
    for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < n; y++) {
            size_t z = x + y;
            for (size_t j = 0; j < BITS; j++) {
                MAT_AT(NN_INPUT(&nn), 0, j) = (x>>j)&1; 
                MAT_AT(NN_INPUT(&nn), 0, j + BITS) = (y>>j)&1;
            }
            
            nn_forward(&nn);
            
            if(MAT_AT(NN_OUTPUT(&nn), 0, BITS) >0.5f) {
                if(z > n) {
                    //printf("%u + %u = (OVERFLOW|%u)\n", x, y, z);
                    //fails++;
                }
            } else {
                size_t a = 0;
                for (size_t j = 0; j < BITS; j++) {
                    size_t bit = MAT_AT(NN_OUTPUT(&nn), 0, j) > 0.5f;
                    a |= bit<<j;
                }
                if(z != a) {
                    printf("%u + %u = (%u|%u)\n", x, y, z, a);
                    fails++;
                }

            }
        }
    }
    if(fails == 0) {
        printf("OK\n");
    }
       
    
    return 0;
}