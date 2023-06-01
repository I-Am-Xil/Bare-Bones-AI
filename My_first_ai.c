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

float rand_float(void)
{
    return (float) rand()/ (float) RAND_MAX;
}

float cost(float w, float b){
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x = train[i][0];
        float y = x*w + b;
        float d = y - train[i][1];

        result += d*d;
    }
    return result /= train_count;
}

int main()
{
    srand(time(0));
    float w = rand_float()*10.0f;
    float b = rand_float()*2.5f;

    float epsilon = 1e-3;
    float rate = 1e-3;

    printf("%f\n", cost(w, b));

    for (size_t i = 0; i < 2000; i++)
    {
        float cost_value = cost(w, b);
        float dw = (cost(w + epsilon, b) - cost_value)/epsilon;
        float db = (cost(w, b + epsilon) - cost_value)/epsilon;
        w -= rate*dw;
        b -= rate*db;

        printf("cost = %f, w = %f, b = %f\n", cost(w, b), w, b);
    }

    printf("-----------------------------------\n");
    printf("w = %f, b = %f\n", w, b);

    return 0;
}
