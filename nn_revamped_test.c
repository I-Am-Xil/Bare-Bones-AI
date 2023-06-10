#include<time.h>
#include<stdlib.h>
#include<stdio.h>
#include "libraries/nn_revamped/nn_revamped.h"

float td[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0
};

int main() {
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;

    float epsilon = 1e-1;
    float rate = 1e-1;
    float power = 2;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2
    };

    size_t arch[] = {2, 1};

    NN nn;
    NN g;

    nn_alloc(&nn, arch, ARRAY_LEN(arch));
    nn_alloc(&g, arch, ARRAY_LEN(arch));
    nn_rand(&nn, 0, 1);

    NN_PRINT(&nn);

    printf("cost = %f\n", nn_nonlinear_cost(&nn, &ti, &to, power));
    for (size_t i = 0; i < 10000; i++) {
        nn_nonlinear_finite_diff(&nn, &g, epsilon, &ti, &to, power);
        nn_learn(&nn, &g, rate);
        //printf("cost = %f\n", nn_cost(nn_pointer, ti_pointer, to_pointer));
    }

    printf("cost = %f\n", nn_nonlinear_cost(&nn, &ti, &to, power));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(&NN_INPUT(&nn), 0, 0) = i;
            MAT_AT(&NN_INPUT(&nn), 0, 1) = j;
            nn_nonlinear_forward(&nn, power);
            printf("%u | %u = %f\n", i, j, MAT_AT(&NN_OUTPUT(&nn), 0, 0));
        }
    }

    return 0;
}