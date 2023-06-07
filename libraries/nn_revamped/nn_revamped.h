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

#define MAT_AT(m_pointer, i, j) m_pointer->es[(i)*(m_pointer)->stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_sig(Mat *m);
void mat_polinomial_transform(Mat *m, float power);
void mat_mul(Mat *dst, Mat *a, Mat *b);
void mat_sum(Mat *dst, Mat *a);
Mat mat_row(Mat *m, size_t row);
void mat_copy(Mat *dst, Mat *src);
void mat_rand(Mat *m, float low, float high);
void mat_fill(Mat *dst, float f);
void mat_print(Mat *m_pointer, const char *name, size_t padding);
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
void nn_print(NN *nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn)
void nn_rand(NN *nn_pointer, float low, float high);
void nn_forward(NN *nn_pointer);
float nn_cost(NN *nn_pointer, Mat *ti, Mat *to);
void nn_finite_diff(NN *nn_pointer, NN *g_pointer, float epsilon, Mat *ti, Mat *to);
void nn_learn(NN *nn_pointer, NN *g_pointer, float rate);

void nn_nonlinear_forward(NN *nn_pointer, float power);
float nn_nonlinear_cost(NN *nn_pointer, Mat *ti, Mat *to, float power);
void nn_nonlinear_finite_diff(NN *nn_pointer, NN *g_pointer, float epsilon, Mat *ti, Mat *to, float power);
#endif //NN_REVAMPED_H_