#include<stdio.h>
#include<stdlib.h>


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

float cost(float w){
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x = train[i][0];
        float y = x*w;
        float d = y - train[i][1];

        result += d*d;
    }
    return result /= train_count;
}

int main()
{
    srand(69);
    float w = rand_float()*10.0f;
    float epsilon = 1e-3;
    float rate = 1e-3;

    printf("%f\n", cost(w));

    for (size_t i = 0; i < 500; i++)
    {
        float dcost = (cost(w + epsilon) - cost(w))/epsilon;
        w -= rate*dcost;
        printf("%f\n", cost(w));
    }
    printf("-----------------------------------\n");
    printf("%f", w);
    
    

    return 0;
}
