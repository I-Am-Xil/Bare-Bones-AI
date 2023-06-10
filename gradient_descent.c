#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define train_count (sizeof(train)/sizeof(train[0]))

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8}
};

float rand_float(void) {
    return (float) rand()/ (float) RAND_MAX;
}

float dcost(float w) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x = train[i][0];
        float y = train[i][1];
        result += 2*(x*w - y)*x;
    }
    return result /= train_count;
}


int main() {
    srand(time(0));
    float w = rand_float()*10.0f;

    float rate = 1e-1;

    printf("%f\n", dcost(w));

    for (size_t i = 0; i < 11; i++) {
        float cost_value = dcost(w);
        float dw = dcost(w);
        w -= rate*dw;

        printf("cost = %f, w = %f\n", dcost(w), w);
    }

    printf("-----------------------------------\n");
    printf("w = %f\n", w);

    return 0;
}