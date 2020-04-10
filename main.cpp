#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "src/exploration.h"
#include "src/network.h"

namespace la = boost::numeric::ublas;


int main() {
//    preparing data
    la::vector<double> target = read_target("../data/mnist_y.txt");
//    std::vector<la::vector<double>> data = read_data("../data/mnist_x.txt");
    std::vector<la::vector<double>> data = read_data("../data/mnist_x_scaled.txt");


//    std::vector<la::vector<double>> scaled_data;

//    for (auto& v: data)
//        scaled_data.emplace_back(std::move(standart_scaler(v)));

    auto [train_x, test_x] = split_set(data, 0.1);
    auto [prev_train_y, test_y] = split_vector(target, 0.1);

    std::vector<la::vector<double>> train_y;
    la::vector<double> val_vector(10);
    for (double val: prev_train_y) {
        for (int i = 0; i < 10; i++) {
            val_vector(i) = 0;
        }
        val_vector((int) val) = 1;
        train_y.push_back(val_vector);
    }




//    training model
    std::vector<int> nn_struct = {64, 30, 10};
    auto [W, b] = train(nn_struct, train_x, train_y, 3000);

    auto pred_y = predict(W, b, test_x, nn_struct.size());
//
    std::cout << pred_y << std::endl;
    std::cout << test_y << std::endl;

    return 0;
}