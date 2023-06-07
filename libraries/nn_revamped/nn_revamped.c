#include<stddef.h>
#include<stdio.h>
#include<math.h>

#ifndef NN_MALLOC
#include<stdlib.h>
#define NN_MALLOC malloc
#endif //NN_MALLOC

#ifndef NN_ASSERT
#include<assert.h>
#define NN_ASSERT assert
#endif //NN_ASSERT

#include "nn_revamped.h"

/*
* alloc
* rand
* finite diff
* learn
* forward
*/

float rand_float(void) {
    return (float) rand()/ (float) RAND_MAX;
}


float sigmoidf(float x) {
    return 1.f/ (1.f + expf(-x));
}


Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = malloc(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);

    return m;
}


void mat_print(Mat *m, const char *name, size_t padding) {
    printf("%*s%s = [\n", (int) padding, "", name);

    for (size_t i = 0; i < m->rows; i++) {
        printf("%*s", (int) padding, "");
        for (size_t j = 0; j < m->cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}


void mat_fill(Mat *dst, float f) {
    for (size_t i = 0; i < dst->rows; i++) {
        for (size_t j = 0; j < dst->cols; j++) {
            MAT_AT(dst, i, j) = f;
        }
    }
}


void mat_rand(Mat *m, float low, float high) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}


void mat_mul(Mat *dst, Mat *a, Mat *b) {
    assert(a->cols == b->rows);
    size_t n = a->cols;

    assert(dst->rows == a->rows);
    assert(dst->cols == b->cols);

    for (size_t i = 0; i < dst->rows; i++) {
        for (size_t j = 0; j < dst->cols; j++) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < n; k++) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k)*MAT_AT(b, k, j);
            }
        }
    }
}


void mat_sum(Mat *dst, Mat *a) {
    assert(dst->cols == a->cols);
    assert(dst->rows == a->rows);

    for (size_t i = 0; i < dst->rows; i++) {
        for (size_t j = 0; j < dst->cols; j++) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}


void mat_sig(Mat *m) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}


void mat_polinomial_transform(Mat *m, float power) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            MAT_AT(m, i, j) = powf(MAT_AT(m, i, j), power);
        }
    }
}


Mat mat_row(Mat *m, size_t row) {
    return (Mat) {
        .rows = 1,
        .cols = m->cols,
        .stride = m->stride,
        .es = &MAT_AT(m, row, 0)
    };
}


void mat_copy(Mat *dst, Mat *src) {
    assert(dst->cols == src->cols);
    assert(dst->rows == src->rows);

    for (size_t i = 0; i < dst->rows; i++) {
        for (size_t j = 0; j < dst->cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}


void nn_alloc(NN *nn_pointer, size_t *arch, size_t arch_count) {

    NN_ASSERT(arch_count > 0);
    nn_pointer->count = arch_count - 1;

    nn_pointer->ws = NN_MALLOC(sizeof(*nn_pointer->ws)*nn_pointer->count);
    NN_ASSERT(nn_pointer->ws != NULL);
    nn_pointer->bs = NN_MALLOC(sizeof(*nn_pointer->bs)*nn_pointer->count);
    NN_ASSERT(nn_pointer->bs != NULL);
    nn_pointer->as = NN_MALLOC(sizeof(*nn_pointer->as)*nn_pointer->count + 1);
    NN_ASSERT(nn_pointer->as != NULL);

    nn_pointer->as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i < arch_count; i++) {
        nn_pointer->ws[i-1] = mat_alloc(nn_pointer->as[i-1].cols, arch[i]);
        nn_pointer->bs[i-1] = mat_alloc(1, arch[i]);
        nn_pointer->as[i]   = mat_alloc(1, arch[i]);
    }
}


void nn_print(NN *nn, const char *name) {
    char buf[256];
    
    printf("%s = [\n", name);
    
    for (size_t i = 0; i < nn->count; i++) {
        Mat *p;

        p = &nn->ws[i];
        snprintf(buf, sizeof(buf), "ws%u", i);
        mat_print(p, buf, 4);

        p = &nn->bs[i];
        snprintf(buf, sizeof(buf), "bs%u", i);
        mat_print(p, buf, 4);
    }

    printf("]\n");   
}


void nn_rand(NN *nn_pointer, float low, float high) {
    for (size_t i = 0; i < nn_pointer->count; i++) {
        Mat *p;
        p = &nn_pointer->ws[i];
        mat_rand(p, low, high);

        p = &nn_pointer->bs[i];
        mat_rand(p, low, high);
    }
}


void nn_forward(NN *nn_pointer) {
    Mat *p;
    Mat *q;
    Mat *l;
    Mat *k;
    for (size_t i = 0; i < nn_pointer->count; i++) {
        p = &nn_pointer->as[i+1];
        q = &nn_pointer->as[i];
        l = &nn_pointer->ws[i];
        k = &nn_pointer->bs[i];

        mat_mul(p, q, l);
        mat_sum(p, k);
        mat_sig(p);
    }
}


float nn_cost(NN *nn_pointer, Mat *ti, Mat *to) {
    assert(ti->rows == to->rows);
    assert(to->cols == NN_OUTPUT(nn_pointer).cols);

    size_t n = ti->rows;
    float c = 0;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        Mat *x_pointer = &x;
        Mat *y_pointer = &y;
        Mat *input = &NN_INPUT(nn_pointer);
        Mat *output = &NN_OUTPUT(nn_pointer);

        mat_copy(input, x_pointer);
        nn_forward(nn_pointer);

        size_t q = to->cols;
        for (size_t j = 0; j < q; j++) {
            float d = MAT_AT(output, 0, j) - MAT_AT(y_pointer, 0, j);
            c += d*d;
        }
    }
    return c/n;
}


void nn_finite_diff(NN *nn_pointer, NN *g_pointer, float epsilon, Mat *ti, Mat *to) {
    float saved;
    float c = nn_cost(nn_pointer, ti, to);

    Mat *ws_p;
    Mat *bs_p;
    Mat *g_p;
    

    for (size_t i = 0; i < nn_pointer->count; i++) {
        for (size_t j = 0; j < nn_pointer->ws[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->ws[i].cols; k++) {
                ws_p = &nn_pointer->ws[i];
                g_p = &g_pointer->ws[i];
                
                saved = MAT_AT(ws_p, j, k);
                MAT_AT(ws_p, j, k) += epsilon;
                MAT_AT(g_p, j, k) = (nn_cost(nn_pointer, ti, to) - c) /epsilon;
                MAT_AT(ws_p, j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn_pointer->bs[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->bs[i].cols; k++) {
                bs_p = &nn_pointer->bs[i];
                g_p = &g_pointer->bs[i];
                saved = MAT_AT(bs_p, j, k);
                MAT_AT(bs_p, j, k) += epsilon;
                MAT_AT(g_p,  j, k) = (nn_cost(nn_pointer, ti, to) - c) /epsilon;
                MAT_AT(bs_p, j, k) = saved;
            }
        }
    }
}


void nn_learn(NN *nn_pointer, NN *g_pointer, float rate) {
    Mat *nn_p;
    Mat *g_p;

    for (size_t i = 0; i < nn_pointer->count; i++) {
        for (size_t j = 0; j < nn_pointer->ws[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->ws[i].cols; k++) {
                nn_p = &nn_pointer->ws[i];
                g_p = &g_pointer->ws[i];
                MAT_AT(nn_p, j, k) -= rate*MAT_AT(g_p, j, k);
            }
        }

        for (size_t j = 0; j < nn_pointer->bs[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->bs[i].cols; k++) {
                nn_p = &nn_pointer->bs[i];
                g_p = &g_pointer->bs[i];
                MAT_AT(nn_p, j, k) -= rate*MAT_AT(g_p, j, k);
            }
        }
    }
}


void nn_nonlinear_forward(NN *nn_pointer, float power) {
    Mat *p;
    Mat *q;
    Mat *l;
    Mat *k;
    for (size_t i = 0; i < nn_pointer->count; i++) {
        p = &nn_pointer->as[i+1];
        q = &nn_pointer->as[i];
        l = &nn_pointer->ws[i];
        k = &nn_pointer->bs[i];

        mat_mul(p, q, l);
        mat_polinomial_transform(p, power);
        mat_sum(p, k);
        mat_sig(p);
        (void) power;
    }
}


float nn_nonlinear_cost(NN *nn_pointer, Mat *ti, Mat *to, float power) {
    assert(ti->rows == to->rows);
    assert(to->cols == NN_OUTPUT(nn_pointer).cols);

    size_t n = ti->rows;
    float c = 0;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        Mat *x_pointer = &x;
        Mat *y_pointer = &y;
        Mat *input = &NN_INPUT(nn_pointer);
        Mat *output = &NN_OUTPUT(nn_pointer);

        mat_copy(input, x_pointer);
        nn_nonlinear_forward(nn_pointer, power);

        size_t q = to->cols;
        for (size_t j = 0; j < q; j++) {
            float d = MAT_AT(output, 0, j) - MAT_AT(y_pointer, 0, j);
            c += d*d;
        }
    }
    return c/n;
}


void nn_nonlinear_finite_diff(NN *nn_pointer, NN *g_pointer, float epsilon, Mat *ti, Mat *to, float power) {
    float saved;
    float c = nn_nonlinear_cost(nn_pointer, ti, to, power);

    Mat *ws_p;
    Mat *bs_p;
    Mat *g_p;
    

    for (size_t i = 0; i < nn_pointer->count; i++) {
        ws_p = &nn_pointer->ws[i];
        g_p = &g_pointer->ws[i];
        
        for (size_t j = 0; j < nn_pointer->ws[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->ws[i].cols; k++) {
                
                saved = MAT_AT(ws_p, j, k);
                MAT_AT(ws_p, j, k) += epsilon;
                MAT_AT(g_p, j, k) = (nn_nonlinear_cost(nn_pointer, ti, to, power) - c) /epsilon;
                MAT_AT(ws_p, j, k) = saved;
            }
        }

        bs_p = &nn_pointer->bs[i];
        g_p = &g_pointer->bs[i];
        
        for (size_t j = 0; j < nn_pointer->bs[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->bs[i].cols; k++) {
                saved = MAT_AT(bs_p, j, k);
                MAT_AT(bs_p, j, k) += epsilon;
                MAT_AT(g_p,  j, k) = (nn_nonlinear_cost(nn_pointer, ti, to, power) - c) /epsilon;
                MAT_AT(bs_p, j, k) = saved;
            }
        }
    }
}