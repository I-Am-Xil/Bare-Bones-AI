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


void nn_alloc(NN *nn, size_t *arch, size_t arch_count) {

    NN_ASSERT(arch_count > 0);
    nn->count = arch_count - 1;

    nn->ws = NN_MALLOC(sizeof(*nn->ws)*nn->count);
    NN_ASSERT(nn->ws != NULL);
    nn->bs = NN_MALLOC(sizeof(*nn->bs)*nn->count);
    NN_ASSERT(nn->bs != NULL);
    nn->as = NN_MALLOC(sizeof(*nn->as)*nn->count + 1);
    NN_ASSERT(nn->as != NULL);

    nn->as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i < arch_count; i++) {
        nn->ws[i-1] = mat_alloc(nn->as[i-1].cols, arch[i]);
        nn->bs[i-1] = mat_alloc(1, arch[i]);
        nn->as[i]   = mat_alloc(1, arch[i]);
    }
}


void nn_print(NN *nn, const char *name) {
    char buf[256];
    
    printf("%s = [\n", name);
    
    for (size_t i = 0; i < nn->count; i++) {
        snprintf(buf, sizeof(buf), "ws%u", i);
        mat_print(&nn->ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%u", i);
        mat_print(&nn->bs[i], buf, 4);
    }

    printf("]\n");   
}


void nn_rand(NN *nn, float low, float high) {
    for (size_t i = 0; i < nn->count; i++) {
        mat_rand(&nn->ws[i], low, high);
        mat_rand(&nn->bs[i], low, high);
    }
}


void nn_fill(NN *nn, float x) {
    for (size_t i = 0; i < nn->count; i++) {
        mat_fill(&nn->ws[i], x);
        mat_fill(&nn->bs[i], x);
    }
}


void nn_forward(NN *nn) {
    for (size_t i = 0; i < nn->count; i++) {
        mat_mul(&nn->as[i+1], &nn->as[i], &nn->ws[i]);
        mat_sum(&nn->as[i+1], &nn->bs[i]);
        mat_sig(&nn->as[i+1]);
    }
}


float nn_cost(NN *nn, Mat *ti, Mat *to) {
    assert(ti->rows == to->rows);
    assert(to->cols == NN_OUTPUT(nn).cols);

    size_t n = ti->rows;
    float c = 0;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(&NN_INPUT(nn), &x);
        nn_forward(nn);

        size_t q = to->cols;
        for (size_t j = 0; j < q; j++) {
            float d = MAT_AT(&NN_OUTPUT(nn), 0, j) - MAT_AT(&y, 0, j);
            c += d*d;
        }
    }
    return c/n;
}


void nn_backprop(NN *nn, NN *g, Mat *ti, Mat *to) {
    NN_ASSERT(ti->rows == to->rows);
    size_t n = ti->rows;
    NN_ASSERT(NN_OUTPUT(nn).cols == to->cols);

    nn_fill(g, 0);
    
    /*
    i = current sample
    l = current layer
    j = current activation
    k = previous activation
    */

    for (size_t i = 0; i < n; i++) {
        Mat p = mat_row(ti, i);
        mat_copy(&NN_INPUT(nn), &p);
        nn_forward(nn);

        for (size_t j = 0; j < nn->count; j++) {
            mat_fill(&g->as[j], 0);
        }
        

        for (size_t j = 0; j < to->cols; j++) {
            MAT_AT(&NN_OUTPUT(g), 0, j) = MAT_AT(&NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }

        for (size_t l = nn->count; l > 0 ; l--) {
            for (size_t j = 0; j < nn->as[l].cols; j++) {
                float a = MAT_AT(&nn->as[l], 0, j);
                float da = MAT_AT(&g->as[l], 0, j);
                MAT_AT(&g->bs[l-1], 0, j) += 2*da*a*(1 - a);

                for (size_t k = 0; k < nn->as[l-1].cols; k++) {
                    // j = weight matrix col
                    // k = weight matrix row
                    float pa = MAT_AT(&nn->as[l-1], 0, k);
                    float w = MAT_AT(&nn->ws[l-1], k, j);
                    MAT_AT(&g->ws[l-1], k, j) += 2*da*a*(1 - a)*pa;
                    MAT_AT(&g->as[l-1], 0, k) += 2*da*a*(1 - a)*w;
                }
            }
        }
    }

    for (size_t i = 0; i < g->count; i++) {
        for (size_t j = 0; j < g->ws[i].rows; j++) {
            for (size_t k = 0; k < g->ws[i].cols; k++) {
                MAT_AT(&g->ws[i], j, k) /= n;
            }
        }
        
        for (size_t j = 0; j < g->bs[i].rows; j++) {
            for (size_t k = 0; k < g->bs[i].cols; k++) {
                MAT_AT(&g->bs[i], j, k) /= n;
            }
        }
    }
}


void nn_finite_diff(NN *nn, NN *g, float epsilon, Mat *ti, Mat *to) {
    float saved;
    float c = nn_cost(nn, ti, to);

    for (size_t i = 0; i < nn->count; i++) {
        for (size_t j = 0; j < nn->ws[i].rows; j++) {
            for (size_t k = 0; k < nn->ws[i].cols; k++) {
                saved = MAT_AT(&nn->ws[i], j, k);
                MAT_AT(&nn->ws[i], j, k) += epsilon;
                MAT_AT(&g->ws[i], j, k) = (nn_cost(nn, ti, to) - c) /epsilon;
                MAT_AT(&nn->ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn->bs[i].rows; j++) {
            for (size_t k = 0; k < nn->bs[i].cols; k++) {
                saved = MAT_AT(&nn->bs[i], j, k);
                MAT_AT(&nn->bs[i], j, k) += epsilon;
                MAT_AT(&g->bs[i],  j, k) = (nn_cost(nn, ti, to) - c) /epsilon;
                MAT_AT(&nn->bs[i], j, k) = saved;
            }
        }
    }
}


void nn_learn(NN *nn, NN *g, float rate) {
    for (size_t i = 0; i < nn->count; i++) {
        for (size_t j = 0; j < nn->ws[i].rows; j++) {
            for (size_t k = 0; k < nn->ws[i].cols; k++) {
                MAT_AT(&nn->ws[i], j, k) -= rate*MAT_AT(&g->ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn->bs[i].rows; j++) {
            for (size_t k = 0; k < nn->bs[i].cols; k++) {
                MAT_AT(&nn->bs[i], j, k) -= rate*MAT_AT(&g->bs[i], j, k);
            }
        }
    }
}


void nn_nonlinear_forward(NN *nn, float power) {
    for (size_t i = 0; i < nn->count; i++) {
        mat_mul(&nn->as[i+1], &nn->as[i], &nn->ws[i]);
        mat_polinomial_transform(&nn->as[i+1], power);
        mat_sum(&nn->as[i+1], &nn->bs[i]);
        mat_sig(&nn->as[i+1]);
    }
}


float nn_nonlinear_cost(NN *nn, Mat *ti, Mat *to, float power) {
    assert(ti->rows == to->rows);
    assert(to->cols == NN_OUTPUT(nn).cols);

    size_t n = ti->rows;
    float c = 0;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(&NN_INPUT(nn), &x);
        nn_nonlinear_forward(nn, power);

        size_t q = to->cols;
        for (size_t j = 0; j < q; j++) {
            float d = MAT_AT(&NN_OUTPUT(nn), 0, j) - MAT_AT(&y, 0, j);
            c += d*d;
        }
    }
    return c/n;
}

// *jumps around because of the epsilon value being static
void nn_nonlinear_finite_diff(NN *nn, NN *g, float epsilon, Mat *ti, Mat *to, float power) {
    float saved;
    float c = nn_nonlinear_cost(nn, ti, to, power);

    for (size_t i = 0; i < nn->count; i++) {
        for (size_t j = 0; j < nn->ws[i].rows; j++) {
            for (size_t k = 0; k < nn->ws[i].cols; k++) {
                saved = MAT_AT(&nn->ws[i], j, k);
                MAT_AT(&nn->ws[i], j, k) += epsilon;
                MAT_AT(&g->ws[i], j, k) = (nn_nonlinear_cost(nn, ti, to, power) - c) /epsilon;
                MAT_AT(&nn->ws[i], j, k) = saved;
            }
        }
        
        for (size_t j = 0; j < nn->bs[i].rows; j++) {
            for (size_t k = 0; k < nn->bs[i].cols; k++) {
                saved = MAT_AT(&nn->bs[i], j, k);
                MAT_AT(&nn->bs[i], j, k) += epsilon;
                MAT_AT(&g->bs[i],  j, k) = (nn_nonlinear_cost(nn, ti, to, power) - c) /epsilon;
                MAT_AT(&nn->bs[i], j, k) = saved;
            }
        }
    }
}


//TODO: try HSIC bottleneck
// * Backprop for nonlinear trandformations before activation doesn't work. i have no idea why.
// * I'll just comment it out and keep using finite differences until i find a way to make it work
// * or i'll ignore backprop to implement other algorithms. I've been here for 20 hours.
// * I give up. If you read this and want to try it. Good luck.

void nn_nonlinear_backprop(NN *nn, NN *g, Mat *ti, Mat *to, float power) {
    NN_ASSERT(ti->rows == to->rows);
    size_t n = ti->rows;
    NN_ASSERT(NN_OUTPUT(nn).cols == to->cols);

    nn_fill(g, 0);
    
    
    //i = current sample
    //l = current layer
    //j = current activation
    //k = previous activation
    

    for (size_t i = 0; i < n; i++) {
        Mat p = mat_row(ti, i);
        mat_copy(&NN_INPUT(nn), &p);
        nn_nonlinear_forward(nn, power);

        for (size_t j = 0; j < nn->count; j++) {
            mat_fill(&g->as[j], 0);
        }
        

        for (size_t j = 0; j < to->cols; j++) {
            MAT_AT(&NN_OUTPUT(g), 0, j) = MAT_AT(&NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }

        for (size_t l = nn->count; l > 0 ; l--) {
            for (size_t j = 0; j < nn->as[l].cols; j++) {
                float a = MAT_AT(&nn->as[l], 0, j);
                float da = MAT_AT(&g->as[l], 0, j);
                float bd = 2*da*a*(1 - a);
                float pd = 2*bd;
                MAT_AT(&g->bs[l-1], 0, j) += bd;

                for (size_t k = 0; k < nn->as[l-1].cols; k++) {
                    // j = weight matrix col
                    // k = weight matrix row
                    float pa = MAT_AT(&nn->as[l-1], 0, k);
                    float w = MAT_AT(&nn->ws[l-1], k, j);
                    float paw = pd*pa*w;
                    MAT_AT(&g->ws[l-1], k, j) += paw*pa;
                    MAT_AT(&g->as[l-1], 0, k) += paw*w;
                }
            }
        }
    }

    for (size_t i = 0; i < g->count; i++) {
        for (size_t j = 0; j < g->ws[i].rows; j++) {
            for (size_t k = 0; k < g->ws[i].cols; k++) {
                MAT_AT(&g->ws[i], j, k) /= n;
            }
        }
        
        for (size_t j = 0; j < g->bs[i].rows; j++) {
            for (size_t k = 0; k < g->bs[i].cols; k++) {
                MAT_AT(&g->bs[i], j, k) /= n;
            }
        }
    }
}