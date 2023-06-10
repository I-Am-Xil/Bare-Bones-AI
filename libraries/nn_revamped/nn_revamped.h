#ifndef NN_REVAMPED_H_
#define NN_REVAMPED_H_

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m)->es[(i)*(m)->stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_sig(Mat *m);
void mat_polinomial_transform(Mat *m, float power);
void mat_mul(Mat *dst, Mat *a, Mat *b);
void mat_sum(Mat *dst, Mat *a);
Mat mat_row(Mat *m, size_t row);
void mat_copy(Mat *dst, Mat *src);
void mat_rand(Mat *m, float low, float high);
void mat_fill(Mat *dst, float f);
void mat_print(Mat *m, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count+1
} NN;

#define NN_INPUT(nn) (nn)->as[0]
#define NN_OUTPUT(nn) (nn)->as[(nn)->count]

void nn_alloc(NN *nn, size_t *arch, size_t arch_count);
void nn_print(NN *nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn)
void nn_rand(NN *nn, float low, float high);
void nn_fill(NN *nn, float x);
void nn_forward(NN *nn);
float nn_cost(NN *nn, Mat *ti, Mat *to);
void nn_backprop(NN *nn, NN *g, Mat *ti, Mat *to);
void nn_finite_diff(NN *nn, NN *g, float epsilon, Mat *ti, Mat *to);
void nn_learn(NN *nn, NN *g, float rate);

void nn_nonlinear_forward(NN *nn, float power);
float nn_nonlinear_cost(NN *nn, Mat *ti, Mat *to, float power);
void nn_nonlinear_finite_diff(NN *nn, NN *g, float epsilon, Mat *ti, Mat *to, float power);

//! HERE BEGINS HELL
/* // * left a comment about why is commented in the function definition.

void nn_nonlinear_backprop(NN *nn, NN *g, Mat *ti, Mat *to, float power);
*/

#endif //NN_REVAMPED_H_