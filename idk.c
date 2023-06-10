#include<stdio.h>
#include<stdlib.h>
#include"libraries/nn_revamped/nn_revamped.h"
#include<time.h>

float td[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0
};


int main() {
    //srand(time(0));
    srand(69);

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;

    
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

    NN nn;
    NN g;

    //float epsilon = 1e-1;
    float rate = 1;
    float power = 2;
    
    size_t arch[] = {2, 2, 1};
    nn_alloc(&nn, arch, ARRAY_LEN(arch));
    nn_alloc(&g, arch, ARRAY_LEN(arch));
    nn_rand(&nn, 0, 1);

    //printf("cost = %f\n", nn_cost(&nn, &ti, &to));
    printf("cost = %f\n", nn_nonlinear_cost(&nn, &ti, &to, power));

    for (size_t i = 0; i < 10000; i++) {
        nn_nonlinear_backprop(&nn, &g, &ti, &to, power);
        //NN_PRINT(g);
        //nn_finite_diff(nn_pointer, g_pointer, epsilon, ti, to);
        nn_learn(&nn, &g, rate);
        printf("cost = %f\n", nn_nonlinear_cost(&nn, &ti, &to, power));
    }

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