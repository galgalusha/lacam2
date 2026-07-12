#pragma once

#include <iostream>
#include <random>

static const uint PRS_NUM_THREADS     = 7;
static uint CEM_NUM_CANDIDATES        = 20;
static int CEM_ELITE_COUNT            = 20;
static double INIT_LAPLACE_SMOOTHING  = 50.0;
static double GEN_LAPLACE_SMOOTHING   = 0.02;
static double BASE_LEARNING_RATE      = 0.25;
static auto LEARNING_RATE_FUNC   = [](int gen, float base) 
                                   { return base * sqrt(150.0 / (150.0 + gen)); };
static float LEARNING_RATE            = LEARNING_RATE_FUNC(0, BASE_LEARNING_RATE);
