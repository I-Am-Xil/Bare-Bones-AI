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
    Mat *ti_pointer = &ti;

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2
    };
    Mat *to_pointer = &to;

    size_t arch[] = {2, 1};

    NN nn;
    NN *nn_pointer = &nn;

    
    NN g;
    NN *g_pointer = &g;

    nn_alloc(nn_pointer, arch, ARRAY_LEN(arch));
    nn_alloc(g_pointer, arch, ARRAY_LEN(arch));
    nn_rand(nn_pointer, 0, 1);

    Mat *input = &NN_INPUT(nn_pointer);
    Mat *output = &NN_OUTPUT(nn_pointer);

    //NN_PRINT(nn_pointer);

    printf("cost = %f\n", nn_nonlinear_cost(nn_pointer, ti_pointer, to_pointer, power));
    for (size_t i = 0; i < 10000; i++) {
        nn_nonlinear_finite_diff(nn_pointer, g_pointer, epsilon, ti_pointer, to_pointer, power);
        nn_learn(nn_pointer, g_pointer, rate);
        //printf("cost = %f\n", nn_cost(nn_pointer, ti_pointer, to_pointer));
    }

    printf("cost = %f\n", nn_nonlinear_cost(nn_pointer, ti_pointer, to_pointer, power));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(input, 0, 0) = i;
            MAT_AT(input, 0, 1) = j;
            nn_nonlinear_forward(nn_pointer, power);
            printf("%u | %u = %f\n", i, j, MAT_AT(output, 0, 0));
        }
    }

    //NN_PRINT(nn_pointer);

    return 0;
}