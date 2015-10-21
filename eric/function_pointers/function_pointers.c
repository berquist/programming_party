#include <stdio.h>

int function_1(int x, int y) {

  return x + y;

}

int function_2(int x, int y) {

  return x - y;

}

int function_3(int x, int y) {

  return x * y;

}

typedef int (*funcptr)(int x, int y);
funcptr determine_function(int flag) {

  funcptr function_ptr;

  if (flag == -1) {
    function_ptr = function_1;
  } else if (flag == -2) {
    function_ptr = function_2;
  } else if (flag == -3) {
    function_ptr = function_3;
  } else {
    function_ptr = NULL;
  }

  return function_ptr;

}

void main() {

  /* int flag = -1; */

  /* printf("flag: %d\n", flag); */

  /* funcptr function_ptr = determine_function(flag); */

  int x = 5;
  int y = 3;

  printf("%i\n", (determine_function(-1))(x, y));
  printf("%i\n", (determine_function(-2))(x, y));
  printf("%i\n", (determine_function(-3))(x, y));
  /* What will happen when this runs? */
  /* printf("%i\n", (determine_function(-4))(x, y)); */

  return;

}
