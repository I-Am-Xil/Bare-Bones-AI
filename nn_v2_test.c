
//* gcc -Wall -Wextra -o nn_v2_test nn_v2_test.c libraries/nn_v2/nn_v2.c

#include<time.h>
#include"libraries/nn_v2/nn_v2.h"

float td[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 1
};


size_t arch[] = {2, 2, 1};


int main() {
    //srand(time(0));

    size_t stride = 3;
    size_t row = sizeof(td)/sizeof(td[0])/stride;

    Mat ti = {
        .rows = row,
        .cols = 2,
        .stride = stride,
        .value = td
    };

    Mat to = {
        .rows = row,
        .cols = 1,
        .stride = stride,
        .value = td+2
    };


    Fcnn nn;
    printf("%u\n", ARRAY_LEN(arch));

    nn_alloc(&nn, arch, ARRAY_LEN(arch));
    nn_rand(&nn, 0, 1);
    NN_PRINT(&nn);

    nn_forward(&nn);

    NN_PRINT(&nn);

    printf("cost = %f\n", nn_cost(&nn, &ti, &to));

    return 0;
}