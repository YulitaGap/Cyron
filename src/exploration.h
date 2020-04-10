//
// Created by Admin on 07.04.2020.
//

#ifndef CYRON_EXPLORATION_H
#define CYRON_EXPLORATION_H

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <fstream>
#include <vector>

namespace la = boost::numeric::ublas;

std::vector<la::vector<double>> read_data(const std::string& path);
la::vector<double> read_target(const std::string& path);

double mean(la::vector<double>& v);
double sd(la::vector<double>& v);
la::vector<double> standart_scaler(la::vector<double> &v);

std::tuple<std::vector<la::vector<double>>, std::vector<la::vector<double>>> split_set(
        std::vector<la::vector<double>>& v, double test_size);
std::tuple<la::vector<double>, la::vector<double>> split_vector(
        la::vector<double>& v, double test_size);

#endif //CYRON_EXPLORATION_H
