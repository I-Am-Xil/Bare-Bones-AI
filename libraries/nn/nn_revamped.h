#ifndef NN_REVAMPED_H_
#define NN_REVAMPED_H_

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

#endif //NN_REVAMPED_H_