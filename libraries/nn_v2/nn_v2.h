#ifndef NN_V2_
#define NN_V2_

#include<stdio.h>
#include<math.h>

#ifndef NN_MALLOC
#include<stdlib.h>
#define NN_MALLOC malloc
#endif //NN_MALLOC

#ifndef ASSERT
#include<assert.h>
#define ASSERT assert
#endif


#define ARRAY_LEN(object) sizeof((object))/sizeof((object)[0])

//------------------------float-----------------------

float randf();
float sigmoidf(float x);
float d_sigmoidf(float x);

//-----------------------matrix-----------------------

typedef struct
{
    size_t cols;
    size_t rows;
    size_t stride;
    float *value;
} Mat;

#define MAT_AT(m, row, col) (m)->value[(row)*(m)->stride + (col)]

void mat_sigmoid(Mat *dst, Mat *m);

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat *m, float value);
void mat_rand(Mat *m, float min, float max);
void mat_print(Mat *m, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)
Mat mat_row(Mat *m, size_t row);
Mat mat_col(Mat *m, size_t col);
void mat_copy(Mat *dst, Mat *src);
void mat_scalar_mult(Mat *m, float x);
void mat_sum(Mat *dst, Mat *m, Mat *g);
void mat_mul(Mat *dst, Mat *m, Mat *g);

//-----------Fully connected neural network-----------

typedef struct
{
    size_t n_layers;
    Mat *as;
    Mat *ws;
    Mat *bs;
} Fcnn;

#define NN_INPUT(nn) (nn)->as[0]
#define NN_OUTPUT(nn) (nn)->as[(nn)->n_layers]

void nn_alloc(Fcnn *nn, size_t *arch, size_t arch_len);
void nn_print(Fcnn *nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn)
void nn_rand(Fcnn *nn, float min, float max);
void nn_fill(Fcnn *nn, float x);
float nn_cost(Fcnn *nn, Mat *input, Mat *output);
void nn_forward(Fcnn *nn);

#endif //NN_V2_