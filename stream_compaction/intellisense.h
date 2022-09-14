#ifdef __INTELLISENSE__
#define KERN_PARAM(x,y)
#include <device_launch_parameters.h>
#else
#define KERN_PARAM(x,y) <<< x,y >>>
#endif