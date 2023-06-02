
/*
Ik this is dumb. Im learning so idfc tbh. In later iterations i'll change the neuron representation
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

typedef struct
{
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
} Xor;


float sigmoidf(float x){
    return 1.f/ (1.f + expf(-x));
}


float forward(Xor *m, float x1, float x2){
    float a = sigmoidf(m->or_w1*x1 + m->or_w2*x2 + m->or_b);
    float b = sigmoidf(m->nand_w1*x1 + m->nand_w2*x2 + m->nand_b);

    return sigmoidf(m->and_w1*a + m->and_w2*b + m->and_b);
}

typedef float sample[3];
sample xor_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}
};

sample *train = xor_train;
size_t train_count = 4;

float cost(Xor *m){
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x_1 = train[i][0];
        float x_2 = train[i][1];
        float y = forward(m, x_1, x_2);
        float d = y - train[i][2];

        result += d*d;
    }
    result /= train_count;
    return result;
}


float rand_float(void)
{
    return (float) rand()/ (float) RAND_MAX;
}

 
Xor rand_xor(){
    Xor m;
    
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.or_b = rand_float();

    m.nand_w1 = rand_float();
    m.nand_w2 = rand_float();
    m.nand_b = rand_float();

    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.and_b = rand_float();

    return m;
}

void print_xor(Xor *m){

    printf("or_w1   = %f\n", m->or_w1);
    printf("or_w2   = %f\n", m->or_w2);
    printf("or_b    = %f\n", m->or_b);

    printf("nand_w1 = %f\n", m->nand_w1);
    printf("nand_w2 = %f\n", m->nand_w2);
    printf("nand_b  = %f\n", m->nand_b);

    printf("and_w1  = %f\n", m->and_w1);
    printf("and_w2  = %f\n", m->and_w2);
    printf("and_b   = %f\n", m->and_b);
    
}


Xor finite_diff(Xor *m, float epsilon){

    Xor g;
    float c = cost(m);
    float saved;

    saved = m->or_w1;
    m->or_w1 += epsilon;
    g.or_w1 = (cost(m) - c)/epsilon;
    m->or_w1 = saved;

    saved = m->or_w2;
    m->or_w2 += epsilon;
    g.or_w2 = (cost(m) - c)/epsilon;
    m->or_w2 = saved;

    saved = m->or_b;
    m->or_b += epsilon;
    g.or_b = (cost(m) - c)/epsilon;
    m->or_b = saved;

    saved = m->nand_w1;
    m->nand_w1 += epsilon;
    g.nand_w1 = (cost(m) - c)/epsilon;
    m->nand_w1 = saved;
    
    saved = m->nand_w2;
    m->nand_w2 += epsilon;
    g.nand_w2 = (cost(m) - c)/epsilon;
    m->nand_w2 = saved;

    saved = m->nand_b;
    m->nand_b += epsilon;
    g.nand_b = (cost(m) - c)/epsilon;
    m->nand_b = saved;
    
    saved = m->and_w1;
    m->and_w1 += epsilon;
    g.and_w1 = (cost(m) - c)/epsilon;
    m->and_w1 = saved;

    saved = m->and_w2;
    m->and_w2 += epsilon;
    g.and_w2 = (cost(m) - c)/epsilon;
    m->and_w2 = saved;

    saved = m->and_b;
    m->and_b += epsilon;
    g.and_b = (cost(m) - c)/epsilon;
    m->and_b = saved;

    return g;
}

Xor *apply_diff(Xor *m, Xor *g, float rate){
    
    m->or_w1 -= g->or_w1*rate;
    m->or_w2 -= g->or_w2*rate;
    m->or_b -= g->or_b*rate;
    m->nand_w1 -= g->nand_w1*rate;
    m->nand_w2 -= g->nand_w2*rate;
    m->nand_b -= g->nand_b*rate;
    m->and_w1 -= g->and_w1*rate;
    m->and_w2 -= g->and_w2*rate;
    m->and_b -= g->and_b*rate;

    return m;

}

int main(){
    srand(time(0));

    float epsilon = 1e-1;
    float rate = 1e-1;

    Xor m = rand_xor();
    Xor *m_pointer = &m;

    Xor g = finite_diff(m_pointer, epsilon);
    Xor *g_pointer = &g;
    
    for (size_t i = 0; i < 100000; i++)
    {
        g = finite_diff(m_pointer, epsilon);
        m_pointer = apply_diff(m_pointer, g_pointer, rate);
        //printf("cost = %f\n", cost(m_pointer));
    }

    printf("cost = %f\n", cost(m_pointer));

    printf("-----------------------------------\n");

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu | %zu = %f\n", i, j, forward(m_pointer, i, j));
        }
        
    }

    printf("-----------------------------------\n");
    printf("\"OR GATE\"\n");

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu | %zu = %f\n", i, j, sigmoidf(m_pointer->or_w1*i + m_pointer->or_w2*j + m_pointer->or_b));
        }
        
    }

    printf("-----------------------------------\n");
    printf("\"AND GATE\"\n");

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu | %zu = %f\n", i, j, sigmoidf(m_pointer->and_w1*i + m_pointer->and_w2*j + m_pointer->and_b));
        }
        
    }

    printf("-----------------------------------\n");
    printf("\"NAND GATE\"\n");

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu | %zu = %f\n", i, j, sigmoidf(m_pointer->nand_w1*i + m_pointer->nand_w2*j + m_pointer->nand_b));
        }
        
    }
    


    return 0;
}