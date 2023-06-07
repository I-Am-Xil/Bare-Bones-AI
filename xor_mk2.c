#include<time.h>

#define NN_IMPLEMENTATION
#include "libraries/nn/nn.h"


typedef struct {
    Mat a0, a1, a2;
    Mat w1, b1;
    Mat w2, b2;
} Xor;


Xor *xor_alloc(Xor *m_pointer) {
    m_pointer->a0 = mat_alloc(1, 2);
    m_pointer->w1 = mat_alloc(2, 2);
    m_pointer->b1 = mat_alloc(1, 2);
    m_pointer->a1 = mat_alloc(1, 2);
    m_pointer->w2 = mat_alloc(2, 1);
    m_pointer->b2 = mat_alloc(1, 1);
    m_pointer->a2 = mat_alloc(1, 1);

    return m_pointer; 
}


void forward_xor(Xor *m_pointer) {
    mat_dot(m_pointer->a1, m_pointer->a0, m_pointer->w1);
    mat_sum(m_pointer->a1, m_pointer->b1);
    mat_sig(m_pointer->a1);

    mat_dot(m_pointer->a2, m_pointer->a1, m_pointer->w2);
    mat_sum(m_pointer->a2, m_pointer->b2);
    mat_sig(m_pointer->a2);
}


float cost(Xor *m_pointer, Mat ti, Mat to) {
    assert(ti.rows == to.rows);
    assert(to.cols == m_pointer->a2.cols);

    size_t n = ti.rows;
    float c = 0;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(m_pointer->a0, x);
        forward_xor(m_pointer);
        
        size_t q = to.cols;
        for (size_t j = 0; j < q; j++) {
            float d = MAT_AT(m_pointer->a2, 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }

    return c/n;
}


void finite_diff(Xor *m_pointer, Xor *g_pointer, float epsilon, Mat ti, Mat to) {
    float saved;
    float c = cost(m_pointer, ti, to);

    for (size_t i = 0; i < m_pointer->w1.rows; i++) {
        for (size_t j = 0; j < m_pointer->w1.cols; j++) {
            saved = MAT_AT(m_pointer->w1, i, j);
            MAT_AT(m_pointer->w1, i, j) += epsilon;
            MAT_AT(g_pointer->w1, i, j) = (cost(m_pointer, ti, to) - c) /epsilon;
            MAT_AT(m_pointer->w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m_pointer->b1.rows; i++) {
        for (size_t j = 0; j < m_pointer->b1.cols; j++) {
            saved = MAT_AT(m_pointer->b1, i, j);
            MAT_AT(m_pointer->b1, i, j) += epsilon;
            MAT_AT(g_pointer->b1, i, j) = (cost(m_pointer, ti, to) - c) /epsilon;
            MAT_AT(m_pointer->b1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m_pointer->w2.rows; i++) {
        for (size_t j = 0; j < m_pointer->w2.cols; j++) {
            saved = MAT_AT(m_pointer->w2, i, j);
            MAT_AT(m_pointer->w2, i, j) += epsilon;
            MAT_AT(g_pointer->w2, i, j) = (cost(m_pointer, ti, to) - c) /epsilon;
            MAT_AT(m_pointer->w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m_pointer->b2.rows; i++) {
        for (size_t j = 0; j < m_pointer->b2.cols; j++) {
            saved = MAT_AT(m_pointer->b2, i, j);
            MAT_AT(m_pointer->b2, i, j) += epsilon;
            MAT_AT(g_pointer->b2, i, j) = (cost(m_pointer, ti, to) - c) /epsilon;
            MAT_AT(m_pointer->b2, i, j) = saved;
        }
    }
    
}


void xor_learn(Xor *m_pointer, Xor *g_pointer, float rate) {
    for (size_t i = 0; i < m_pointer->w1.rows; i++) {
        for (size_t j = 0; j < m_pointer->w1.cols; j++) {
            MAT_AT(m_pointer->w1, i, j) -= rate*MAT_AT(g_pointer->w1, i, j);
        }
    }

    for (size_t i = 0; i < m_pointer->b1.rows; i++) {
        for (size_t j = 0; j < m_pointer->b1.cols; j++) {
            MAT_AT(m_pointer->b1, i, j) -= rate*MAT_AT(g_pointer->b1, i, j);
        }
    }

    for (size_t i = 0; i < m_pointer->w2.rows; i++) {
        for (size_t j = 0; j < m_pointer->w2.cols; j++) {
            MAT_AT(m_pointer->w2, i, j) -= rate*MAT_AT(g_pointer->w2, i, j);
        }
    }

    for (size_t i = 0; i < m_pointer->b2.rows; i++) {
        for (size_t j = 0; j < m_pointer->b2.cols; j++) {
            MAT_AT(m_pointer->b2, i, j) -= rate*MAT_AT(g_pointer->b2, i, j);
        }
    }
}


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

    Xor m;
    Xor *m_pointer = &m;
    m_pointer = xor_alloc(m_pointer);

    Xor g;
    Xor *g_pointer = &g;
    g_pointer = xor_alloc(g_pointer);

    mat_rand(m_pointer->w1, 0, 1);
    mat_rand(m_pointer->b1, 0, 1);
    mat_rand(m_pointer->w2, 0, 1);
    mat_rand(m_pointer->b2, 0, 1);

    //printf("cost = %f\n", cost(m_pointer, ti, to));
    //printf("%f\n", cost(m_pointer, ti, to));

    float epsilon = 1e-1;
    float rate = 1e-1;

    for (size_t i = 0; i < 100000; i++) {
        finite_diff(m_pointer, g_pointer, epsilon, ti, to);
        xor_learn(m_pointer, g_pointer, rate);
        //printf("%u: cost = %f\n",i , cost(m_pointer, ti, to));
        //printf("%f\n", cost(m_pointer, ti, to));
    }
    
    printf("-------------------------------------------\n");

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(m_pointer->a0, 0, 0) = i;
            MAT_AT(m_pointer->a0, 0, 1) = j;
            forward_xor(m_pointer);
            float y = *m_pointer->a2.es;
            printf("%u | %u = %f\n", i, j, y);
        }       
    }

    return 0;
}