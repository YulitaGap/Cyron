//
// Created by Admin on 07.04.2020.
//

#include "exploration.h"


std::vector<la::vector<double>> read_data(const std::string& path) {
    std::vector<la::vector<double>> result;
    std::ifstream file(path);

    if(!file){
        std::cerr << "Error opening input file" << std::endl;
        exit(1);
    }
    double x;
    std::string line;
    while( getline(file, line) ) {
        std::vector<double> v;
        std::istringstream iss(line);
        for (double s; iss >> s;)
            v.push_back(s);
        la::vector<double> u(v.size());
        for (int i = 0; i < v.size(); i++)
            u(i) = v[i];
        result.emplace_back(std::move(u));
    }
    file.close();

    return result;
}

la::vector<double> read_target(const std::string& path) {
    std::vector<double> v;
    std::ifstream file(path);

    if(!file){
        std::cerr << "Error opening input file" << std::endl;
        exit(1);
    }
    double x;
    while( file >> x )
        v.push_back(x);
    file.close();

    la::vector<double> result(v.size());
    for (int i = 0; i < v.size(); i++)
        result(i) = v[i];

    return result;
}


double mean(la::vector<double>& v) {
    double sum = 0;
    for (auto& el: v)
        sum += el;
    return sum / v.size();
}

double sd(la::vector<double>& v) {
    double u = mean(v);
    double sum = 0;
    for (auto& el: v)
        sum += (el - u) * (el - u);
    return sqrt(sum / v.size());
}

la::vector<double> standart_scaler(la::vector<double> &v) {
    la::vector<double> result(v.size());
    double u = mean(v);
    double s = sd(v);
    for (int i = 0; i < v.size(); ++i)
        result(i) = (v(i) - u) / s;
    return result;
}


std::tuple<std::vector<la::vector<double>>, std::vector<la::vector<double>>> split_set(
        std::vector<la::vector<double>>& v, double test_size) {
    if (test_size > 1 || test_size < 0)
        test_size = 0.2;
    int limit = floor(v.size() * (1 - test_size));

    std::vector<la::vector<double>> train;
    for (int i = 0; i < limit; i++)
        train.emplace_back(std::move(v[i]));

    std::vector<la::vector<double>> test;
    for (int i = limit; i < v.size(); i++)
        test.emplace_back(std::move(v[i]));

    return {train, test};
}

std::tuple<la::vector<double>, la::vector<double>> split_vector(
        la::vector<double>& v, double test_size) {
    if (test_size > 1 || test_size < 0)
        test_size = 0.2;
    int limit = floor(v.size() * (1 - test_size));

    la::vector<double> train(limit);
    for (int i = 0; i < limit; i++)
        train(i) = v(i);

    int length = v.size() - limit;
    la::vector<double> test(length);
    for (int i = limit; i < v.size(); i++)
        test(i - limit) = v(i);

    return {train, test};
}
