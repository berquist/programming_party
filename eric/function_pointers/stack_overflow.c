#include <stdio.h>

int addInt(int n, int m) {
  return n + m;
}

int add2to3(int (*functionPtr)(int, int)) {
  return (*functionPtr)(2, 3);
}

int (*functionFactory(int n))(int, int) {
  printf("Got parameter %d", n);
  int (*functionPtr)(int,int) = &addInt;
  return functionPtr;
}

void main() {

  int (*functionPtr)(int, int);
  /* can also be functionPtr = addInt */
  functionPtr = &addInt;
  int sum = (*functionPtr)(7, 20);
  printf("sum: %d\n", sum);

  int another_sum = add2to3(functionPtr);
  printf("another_sum: %d\n", another_sum);

  return;
}
