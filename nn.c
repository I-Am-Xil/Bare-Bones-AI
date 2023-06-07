#include<time.h>

#define NN_IMPLEMENTATION
#include "libraries/nn/nn.h"

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
    NN *nn_pointer = &nn;

    NN g;
    NN *g_pointer = &g;

    size_t arch[] = {2, 2, 1};
    nn_pointer = nn_alloc(nn_pointer, arch, ARRAY_LEN(arch));
    g_pointer = nn_alloc(g_pointer, arch, ARRAY_LEN(arch));
    nn_rand(nn_pointer, 0, 1);

    //printf("cost = %f\n", nn_cost(nn_pointer, ti, to));

    for (size_t i = 0; i < 50000; i++) {
        nn_finite_diff(nn_pointer, g_pointer, epsilon, ti, to);
        nn_learn(nn_pointer, g_pointer, rate);
        //printf("cost = %f\n", nn_cost(nn_pointer, ti, to));
    }

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(NN_INPUT(nn_pointer), 0, 0) = i;
            MAT_AT(NN_INPUT(nn_pointer), 0, 1) = j;
            nn_forward(nn_pointer);
            printf("%u | %u = %f\n", i, j, MAT_AT(NN_OUTPUT(nn_pointer), 0, 0));
        }
    }
    
    

    return 0;
}