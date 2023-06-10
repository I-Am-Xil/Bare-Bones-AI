#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#define train_count (sizeof(train)/sizeof(train[0]))


float sigmoidf(float x) {
    return 1.f/ (1.f + expf(-x));
}

float parabolaf(float x) {
    return x*x;
}


//XOR-gate
float train[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};


float rand_float(void) {
    return (float) rand()/ (float) RAND_MAX;
}


float cost(float w_1, float w_2, float b) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x_1 = train[i][0];
        float x_2 = train[i][1];
        float y = sigmoidf(parabolaf(x_1*w_1 + x_2*w_2) + b);
        float d = y - train[i][2];

        result += d*d;
    }
    return result /= train_count;
}


void dcost(float w_1, float w_2, float b, float *dw_1, float *dw_2, float *db) {
    *dw_1 = 0;
    *dw_2 = 0;
    *db = 0;

    size_t n = train_count;
    for (size_t i = 0; i < n; i++) {
        float xi = train[i][0];        
        float yi = train[i][1];
        float zi = train[i][2];

        float ai = sigmoidf(xi*w_1 + yi*w_2 + b);
        float di = 2*(ai - zi)*ai*(1 - ai);
        *dw_1 += di*xi;
        *dw_2 += di*yi;
        *db += di;
    }

    *dw_1 /= n;
    *dw_2 /= n;
    *db /= n;
}


void ncost(float w_1, float w_2, float b, float *dw_1, float *dw_2, float *db) {
    *dw_1 = 0;
    *dw_2 = 0;
    *db = 0;

    size_t n = train_count;
    for (size_t i = 0; i < n; i++) {
        float xi = train[i][0];        
        float yi = train[i][1];
        float zi = train[i][2];

        float p = xi*w_1 + yi*w_2;
        float ai = sigmoidf(parabolaf(p) + b);
        float di = 2*(ai - zi)*ai*(1 - ai);
        float dq = 2*di;
        float dp = dq*p;
        
        *dw_1 += dp*xi;
        *dw_2 += dp*yi;
        *db += di;
    }

    *dw_1 /= n;
    *dw_2 /= n;
    *db /= n;
}


int main() {

    srand(time(0));
    float w_1 = rand_float();
    float w_2 = rand_float();
    float b = rand_float();

    float rate = 1e-1;

    for (size_t i = 0; i < 10000; i++) {
        float dw_1, dw_2, db;
        ncost(w_1, w_2, b, &dw_1, &dw_2, &db);
        float cost_value = cost(w_1, w_2, b);

        w_1 -= rate*dw_1;
        w_2 -= rate*dw_2;
        b -= rate*db;

        printf("w_1 = %f, w_2 = %f, b = %f, c = %f\n", w_1, w_2, b, cost_value);
    }
    
    printf("-----------------------------------\n");
    printf("w_1 = %f, w_2 = %f, b = %f\n\n", w_1, w_2, b);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            printf("%u | %u = %f\n", i, j, sigmoidf(parabolaf(i*w_1 + j*w_2) + b));
        }
    }

    return 0;
}