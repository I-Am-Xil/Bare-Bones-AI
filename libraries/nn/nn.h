#ifndef NN_H_
#define NN_H_

#include<stddef.h>
#include<stdio.h>

#ifndef NN_MALLOC
#include<stdlib.h>
#define NN_MALLOC malloc
#endif //NN_MALLOC

#ifndef NN_ASSERT
#include<assert.h>
#define NN_ASSERT assert
#endif //NN_ASSERT

#ifndef NN_MATH
#include<math.h>
#define NN_MATH math
#endif //NN_MATH


#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_sig(Mat m);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat dst, float f);
void mat_print(Mat m, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count+1
} NN;

#define NN_INPUT(nn_pointer) (nn_pointer)->as[0]
#define NN_OUTPUT(nn_pointer) (nn_pointer)->as[(nn_pointer)->count]

void nn_alloc(NN *nn_pointer, size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn)
void nn_rand(NN *nn_pointer, float low, float high);
void nn_forward(NN *nn_pointer);
float nn_cost(NN *nn_pointer, Mat ti, Mat to);
void nn_finite_diff(NN *nn_pointer, NN *g_pointer, float epsilon, Mat ti, Mat to);
void nn_backprop(NN *nn_pointer, NN *g_pointer, Mat ti, Mat to);
void nn_learn(NN *nn_pointer, NN *g_pointer, float rate);
#endif //NN_H_


#ifdef NN_IMPLEMENTATION

float rand_float(void) {
    return (float) rand()/ (float) RAND_MAX;
}


Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = malloc(sizeof(*m.es)*rows*cols);
    assert(m.es != NULL);
    return m;
}


void mat_dot(Mat dst, Mat a, Mat b) {
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;

    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i ,j) = 0;
            for (size_t k = 0; k < n; k++) {
                MAT_AT(dst, i ,j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}


Mat mat_row(Mat m, size_t row) {
    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)
    };
}


void mat_copy(Mat dst, Mat src) {
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}


void mat_sum(Mat dst, Mat a) {
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}


void mat_print(Mat m, const char *name, size_t padding) {

    printf("%*s%s = [\n", (int) padding, "", name);

    for (size_t i = 0; i < m.rows; i++) {
        printf("%*s", (int) padding, "");
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}


void mat_rand(Mat m, float low, float high) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}


void mat_fill(Mat dst, float f) {
    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) = f;
        }
    }
}


float sigmoidf(float x) {
    return 1.f/ (1.f + expf(-x));
}


void mat_sig(Mat m) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}


/*
size_t arch[] = {2, 2, 1};

NN nn = nn_alloc(arch, ARRAY_LEN(arch));
NN *nn_pointer = &nn;
nn_pointer = nn_alloc(arch, ARRAY_LEN(arch)); 
*/


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


void nn_print(NN nn, const char *name) {
    char buf[256];
    
    printf("%s = [\n", name);
    
    for (size_t i = 0; i < nn.count; i++) {
        snprintf(buf, sizeof(buf), "ws%u", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%u", i);
        mat_print(nn.bs[i], buf, 4);
    }

    printf("]\n");
    
}


void nn_rand(NN *nn_pointer, float low, float high) {
    for (size_t i = 0; i < nn_pointer->count; i++) {
        mat_rand(nn_pointer->ws[i], low, high);
        mat_rand(nn_pointer->bs[i], low, high);
    }
}


void nn_fill(NN *nn_pointer, float x) {
    for (size_t i = 0; i < nn_pointer->count; i++) {
        mat_fill(nn_pointer->ws[i], x);
        mat_fill(nn_pointer->bs[i], x);
    }
}


void nn_forward(NN *nn_pointer) {
    for (size_t i = 0; i < nn_pointer->count; i++) {
        mat_dot(nn_pointer->as[i+1], nn_pointer->as[i], nn_pointer->ws[i]);
        mat_sum(nn_pointer->as[i+1], nn_pointer->bs[i]);
        mat_sig(nn_pointer->as[i+1]);
    }
}


float nn_cost(NN *nn_pointer, Mat ti, Mat to) {
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn_pointer).cols);

    size_t n = ti.rows;
    float c = 0;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(NN_INPUT(nn_pointer), x);
        nn_forward(nn_pointer);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++) {
            float d = MAT_AT(NN_OUTPUT(nn_pointer), 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }
    return c/n;
}


void nn_backprop(NN *nn_pointer, NN *g_pointer, Mat ti, Mat to) {
    NN_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    NN_ASSERT(NN_OUTPUT(nn_pointer).cols == to.cols);

    nn_fill(g_pointer, 0);
    
    /*
    i = current sample
    l = current layer
    j = current activation
    k = previous activation
    */

    for (size_t i = 0; i < n; i++) {
        mat_copy(NN_INPUT(nn_pointer), mat_row(ti, i));
        nn_forward(nn_pointer);

        for (size_t j = 0; j < nn_pointer->count; j++) {
            mat_fill(g_pointer->as[j], 0);
        }
        

        for (size_t j = 0; j < to.cols; j++) {
            MAT_AT(NN_OUTPUT(g_pointer), 0, j) = MAT_AT(NN_OUTPUT(nn_pointer), 0, j) - MAT_AT(to, i, j);
        }

        for (size_t l = nn_pointer->count; l > 0 ; l--) {
            for (size_t j = 0; j < nn_pointer->as[l].cols; j++) {
                float a = MAT_AT(nn_pointer->as[l], 0, j);
                float da = MAT_AT(g_pointer->as[l], 0, j);
                MAT_AT(g_pointer->bs[l-1], 0, j) += 2*da*a*(1 - a);

                for (size_t k = 0; k < nn_pointer->as[l-1].cols; k++) {
                    // j = weight matrix col
                    // k = weight matrix row
                    float pa = MAT_AT(nn_pointer->as[l-1], 0, k);
                    float w = MAT_AT(nn_pointer->ws[l-1], k, j);
                    MAT_AT(g_pointer->ws[l-1], k, j) += 2*da*a*(1 - a)*pa;
                    MAT_AT(g_pointer->as[l-1], 0, k) += 2*da*a*(1 - a)*w;
                }
            }
        }
    }

    for (size_t i = 0; i < g_pointer->count; i++) {
        for (size_t j = 0; j < g_pointer->ws[i].rows; j++) {
            for (size_t k = 0; k < g_pointer->ws[i].cols; k++) {
                MAT_AT(g_pointer->ws[i], j, k) /= n;
            }
        }
        
        for (size_t j = 0; j < g_pointer->bs[i].rows; j++) {
            for (size_t k = 0; k < g_pointer->bs[i].cols; k++) {
                MAT_AT(g_pointer->bs[i], j, k) /= n;
            }
        }
    }
}


void nn_finite_diff(NN *nn_pointer, NN *g_pointer, float epsilon, Mat ti, Mat to) {
    float saved;
    float c = nn_cost(nn_pointer, ti, to);

    for (size_t i = 0; i < nn_pointer->count; i++) {
        for (size_t j = 0; j < nn_pointer->ws[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->ws[i].cols; k++) {
                saved = MAT_AT(nn_pointer->ws[i], j, k);
                MAT_AT(nn_pointer->ws[i], j, k) += epsilon;
                MAT_AT(g_pointer->ws[i], j, k) = (nn_cost(nn_pointer, ti, to) - c) /epsilon;
                MAT_AT(nn_pointer->ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn_pointer->bs[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->bs[i].cols; k++) {
                saved = MAT_AT(nn_pointer->bs[i], j, k);
                MAT_AT(nn_pointer->bs[i], j, k) += epsilon;
                MAT_AT(g_pointer->bs[i],  j, k) = (nn_cost(nn_pointer, ti, to) - c) /epsilon;
                MAT_AT(nn_pointer->bs[i], j, k) = saved;
            }
        }
    }
}


void nn_learn(NN *nn_pointer, NN *g_pointer, float rate) {
    for (size_t i = 0; i < nn_pointer->count; i++) {
        for (size_t j = 0; j < nn_pointer->ws[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->ws[i].cols; k++) {
                MAT_AT(nn_pointer->ws[i], j, k) -= rate*MAT_AT(g_pointer->ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn_pointer->bs[i].rows; j++) {
            for (size_t k = 0; k < nn_pointer->bs[i].cols; k++) {
                MAT_AT(nn_pointer->bs[i], j, k) -= rate*MAT_AT(g_pointer->bs[i], j, k);
            }
        }
    }
}
#endif //NN_IMPLEMENTATION