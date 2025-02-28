#ifndef KERNELS_H
#define KERNELS_H

void launchVecAdd(float *a , float *b , float *c , int size);

void launchShared(float *a, float *b, float *c , int size);

void launchCoalesced(float *a , float *b , float *c , int size);

void launchTiled(float *a , float *b , float *c , int size);

void launchMultiElement(float *a , float *b , float *c , int size);

void launchVectorized(float *a , float *b , float *c , int size);

#endif