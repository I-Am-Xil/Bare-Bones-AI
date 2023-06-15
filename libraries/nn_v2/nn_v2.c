#include"nn_v2.h"

//-----------------float operations-----------------

float randf() {
    return (float) rand() / (float) RAND_MAX;
}


float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}


float d_sigmoidf(float x) {
    return sigmoidf(x) * (1.f - sigmoidf(x));
}

//---------------activation funcitons---------------

void mat_sigmoid(Mat *dst, Mat *m) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            MAT_AT(dst, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

//-------------common matrix operations-------------

Mat mat_alloc(size_t rows, size_t cols) {
    ASSERT(cols != 0);
    ASSERT(rows != 0);

    Mat m = {
        m.cols=cols,
        m.rows=rows,
        m.stride=cols,
        m.value=malloc(sizeof(*m.value)*rows*cols)
    };

    return m;
}


void mat_fill(Mat *m, float value) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            MAT_AT(m, i, j) = value;
        }
    }
}


void mat_rand(Mat *m, float min, float max) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            MAT_AT(m, i, j) = randf()*(max - min) + min;
        }
    }
}


void mat_print(Mat *m, const char *name, size_t padding) {
    printf("%*s%s = [\n", (int) padding, "", name);

    for (size_t i = 0; i < m->rows; i++) {
        printf("%*s" , (int) padding, "");
        for (size_t j = 0; j < m->cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}


Mat mat_row(Mat *m, size_t row) {

    ASSERT(m->rows > row);

    return (Mat) {
        .cols=m->cols,
        .rows=1,
        .stride=m->stride,
        .value=&MAT_AT(m, row, 0)
    };
}


Mat mat_col(Mat *m, size_t col) {
    
    ASSERT(m->cols > col);

    return (Mat) {
        .cols=1,
        .rows=m->rows,
        .stride=m->stride,
        .value=&MAT_AT(m, 0, col)
    };
}


void mat_copy(Mat *dst, Mat *src) {
    ASSERT(dst->cols == src->cols);
    ASSERT(dst->rows == src->rows);

    for (size_t i = 0; i < src->rows; i++) {
        for (size_t j = 0; j < src->cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}


void mat_scalar_mult(Mat *m, float x) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            MAT_AT(m, i, j) *= x;
        }
    }
}


void mat_sum(Mat *dst, Mat *m, Mat *g) {

    ASSERT(m->cols == g->cols);
    ASSERT(m->rows == g->rows);
    ASSERT(dst->cols == g->cols);
    ASSERT(dst->rows == g->rows);

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(m, i, j) + MAT_AT(g, i, j);
        }
    }
}

void mat_mul(Mat *dst, Mat *m, Mat *g) {
    ASSERT(m->cols == g->rows);
    ASSERT(m->rows == g->cols);
    ASSERT(dst->rows == m->rows);
    ASSERT(dst->cols == g->cols);

    for (size_t i = 0; i < dst->rows; i++) {
        for (size_t j = 0; j < dst->cols; j++) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < dst->cols; k++) {
                MAT_AT(dst, i, j) += MAT_AT(m, i, k) * MAT_AT(g, k, j);
            }
        }
    }
}


//-----------Fully connected neural network-----------

void nn_alloc(Fcnn *nn, size_t *arch, size_t arch_len) {
    
    ASSERT(arch_len > 0);
    nn->n_layers = arch_len - 1;

    nn->ws = NN_MALLOC(sizeof(*nn->ws)*nn->n_layers);
    ASSERT(nn->ws != NULL);
    nn->bs = NN_MALLOC(sizeof(*nn->bs)*nn->n_layers);
    ASSERT(nn->bs != NULL);
    nn->as = NN_MALLOC(sizeof(*nn->as)*nn->n_layers + 1);
    ASSERT(nn->as != NULL);

    nn->as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i < arch_len; i++) {
        nn->ws[i-1] = mat_alloc(nn->as[i-1].cols, arch[i]);
        nn->bs[i-1] = mat_alloc(1, arch[i]);
        nn->as[i]   = mat_alloc(1, arch[i]);
    }
}


void nn_print(Fcnn *nn, const char *name) {
    char buf[256];

    printf("%s = [\n", name);
    for (size_t i = 0; i < nn->n_layers; i++) {
        snprintf(buf, sizeof(buf), "ws%u", i);
        mat_print(&nn->ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%u", i);
        mat_print(&nn->bs[i], buf, 4);
    }
    printf("]\n");
}


void nn_rand(Fcnn *nn, float min, float max) {
    for (size_t i = 0; i < nn->n_layers; i++) {
        mat_rand(&nn->ws[i], min, max);
        mat_rand(&nn->bs[i], min, max);
    }
}


void nn_fill(Fcnn *nn, float x) {
    for (size_t i = 0; i < nn->n_layers; i++) {
        mat_fill(&nn->ws[i], x);
        mat_fill(&nn->bs[i], x);
    }
}


void nn_forward(Fcnn *nn) {
    for (size_t i = 1; i < nn->n_layers; i++) {
        mat_mul(&nn->as[i+1], &nn->as[i], &nn->ws[i]);
        mat_sum(&nn->as[i+1], &nn->as[i+1], &nn->bs[i]);
        mat_sigmoid(&nn->as[i+1], &nn->as[i+1]);
    }
}


float nn_cost(Fcnn *nn, Mat *input, Mat *output) {
    ASSERT(input->rows == output->rows);
    ASSERT(output->cols == NN_OUTPUT(nn).cols);
    size_t n = input->rows;

    float cost = 0.f;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(input, i);
        Mat y = mat_row(output, i);

        mat_copy(&NN_INPUT(nn), &x);
        nn_forward(nn);

        size_t q = input->cols;
        for (size_t j = 0; j < q; j++) {
            float d = MAT_AT(&NN_OUTPUT(nn), 0, j) - MAT_AT(&y, 0, j);
            cost += d*d;
        }
    }
    return (float) (cost) / (float) n;
}


void nn_backprop(Fcnn *nn, Fcnn *delta, Mat *input, Mat *output) {
    ASSERT(input->rows == output->rows);
    ASSERT(output->cols == NN_OUTPUT(nn).cols);
    size_t n = input->rows;

    nn_fill(delta, 0);
    for (size_t i = 0; i < nn->n_layers; i++) {
        mat_fill(&delta->as[i], 0);
    }
    

    for (size_t i_row = 0; i_row < n; i_row++) {

        Mat p = mat_row(input, i_row);
        mat_copy(&NN_INPUT(nn), &p);

        for (size_t l = nn->n_layers; l > 0; l--) {
            if(l == nn->n_layers) {

            }
        }
    }
}

