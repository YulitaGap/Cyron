//
// Created by Admin on 09.04.2020.
//

#ifndef CYRON_NETWORK_H
#define CYRON_NETWORK_H

#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include <map>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <fstream>
#include <vector>

namespace la = boost::numeric::ublas;

la::vector<double> sigmoid(la::vector<double> x);
la::vector<double> sigmoid_d(la::vector<double> x);

la::matrix<double> generateMatrix(int m, int n, std::string type);
la::vector<double> generateVector(int size, std::string type);

std::tuple<std::map<int, la::matrix<double>>, std::map<int, la::vector<double>>> init_weights(std::vector<int>& nn_struct);
std::tuple<std::map<int, la::matrix<double>>, std::map<int, la::vector<double>>> init_tri_weights(std::vector<int>& nn_struct);

std::tuple<std::map<int, la::vector<double>>, std::map<int, la::vector<double>>> feed_forward(
        la::vector<double>& x, std::map<int, la::matrix<double>>& W, std::map<int, la::vector<double>>& b);

la::vector<double> calculate_out_layer_delta(la::vector<double>& y, la::vector<double>& h_out, la::vector<double>& z_out);
la::vector<double> calculate_hidden_delta(la::vector<double>& delta_plus_1, la::matrix<double>& w_l, la::vector<double>& z_l);

std::tuple<std::map<int, la::matrix<double>>, std::map<int, la::vector<double>>> train(
        std::vector<int>& nn_struct, std::vector<la::vector<double>>& x, std::vector<la::vector<double>>& y,
        int iter_num=1000, double alpha = 0.25);

la::vector<double> predict(std::map<int, la::matrix<double>>& W, std::map<int, la::vector<double>>& b,
                                        std::vector<la::vector<double>>& x, int num_layers);

#endif //CYRON_NETWORK_H
