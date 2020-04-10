//
// Created by Admin on 09.04.2020.
//



#include "network.h"

la::vector<double> sigmoid(la::vector<double> x) {
    la::vector<double> result(x.size());
    for (int i = 0; i < x.size(); i++)
        result(i) = 1 / (1 + exp(-x(i)));
    return result;
}

la::vector<double> sigmoid_d(la::vector<double> x) {
    la::vector<double> result(x.size());
    for (int i = 0; i < x.size(); i++)
        result(i) = (1 / (1 + exp(-x(i))) * (1 - 1 / (1 + exp(-x(i)))));
    return result;
}

la::matrix<double> generateMatrix(int m, int n, std::string type) {
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> distribution(1,99);
    la::matrix<double> A(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            if (type == "random")
                A(i, j) = distribution(generator) / 100.00;
            else
                A(i, j) = 0.0;
        }
    return A;
}

la::vector<double> generateVector(int size, std::string type) {
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> distribution(1,99);
    la::vector<double> v(size);
    for (int i = 0; i < size; i++) {
        if (type == "random")
            v(i) = distribution(generator) / 100.00;
        else
            v(i) = 0.0;
    }
    return v;
}

std::tuple<std::map<int, la::matrix<double>>, std::map<int, la::vector<double>>> init_weights(std::vector<int>& nn_struct) {
    std::map<int, la::matrix<double>> W;
    std::map<int, la::vector<double>> b;
    std::string type = "random";
    for (int i = 1; i < nn_struct.size(); i++) {
        W[i] = generateMatrix(nn_struct[i], nn_struct[i-1], type);
        b[i] = generateVector(nn_struct[i], type);
    }
    return {W, b};
}

std::tuple<std::map<int, la::matrix<double>>, std::map<int, la::vector<double>>> init_tri_weights(std::vector<int>& nn_struct) {
    std::map<int, la::matrix<double>> W;
    std::map<int, la::vector<double>> b;
    std::string type = "zero";
    for (int i = 1; i < nn_struct.size(); i++) {
        W[i] = generateMatrix(nn_struct[i], nn_struct[i-1], type);
        b[i] = generateVector(nn_struct[i], type);
    }
    return {W, b};
}


std::tuple<std::map<int, la::vector<double>>, std::map<int, la::vector<double>>> feed_forward(
        la::vector<double>& x, std::map<int, la::matrix<double>>& W, std::map<int, la::vector<double>>& b) {
    std::map<int, la::vector<double>> h;
    h[1] = x;
    std::map<int, la::vector<double>> z;
    la::vector<double> node_in = x;
    for (int i = 1; i < W.size() + 1; i++) {
        if (i > 1)
            node_in = h[i];
        z[i+1] = la::prod(W[i], node_in) + b[i];
        h[i+1] = sigmoid(z[i+1]);
    }
    return {h, z};
}


la::vector<double> calculate_out_layer_delta(la::vector<double>& y, la::vector<double>& h_out, la::vector<double>& z_out) {
    la::vector<double> result = h_out - y;
    la::vector<double> sigm = sigmoid_d(z_out);
    for (int i = 0; i < result.size(); i++)
        result(i) *= sigm(i);
    return result;
}

la::vector<double> calculate_hidden_delta(la::vector<double>& delta_plus_1, la::matrix<double>& w_l, la::vector<double>& z_l){
    la::vector<double> result = la::prod(la::trans(w_l), delta_plus_1);
    la::vector<double> sigm_d = sigmoid_d(z_l);
    for (int i = 0; i < result.size(); i++)
        result(i) *= sigm_d(i);
    return result;
}


std::tuple<std::map<int, la::matrix<double>>, std::map<int, la::vector<double>>> train(
        std::vector<int>& nn_struct, std::vector<la::vector<double>>& x, std::vector<la::vector<double>>& y,
        int iter_num, double alpha) {
    auto [W, b] = init_weights(nn_struct);
    int cnt = 0;
    int m = y.size();
    while (cnt < iter_num) {
        if (cnt % 10 == 0) std::cout << "Iteration " << cnt << std:: endl;
        auto [tri_W, tri_b] = init_tri_weights(nn_struct);

        for (int i = 0; i < y.size(); i++) {
            std::map<int, la::vector<double>> delta;
            auto [h, z] = feed_forward(x[i], W, b);

            for (int l = nn_struct.size(); l < 0; l--) {
                if (l == nn_struct.size())
                    delta[l] = calculate_out_layer_delta(y[i], h[l], z[l]);
                else {
                    if (l > 1)
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l]);
                    tri_W[l] += la::outer_prod(delta[l+1], h[l]);
                    tri_b[l] += delta[l+1];
                }
            }
        }
//        gradient descending
        for (int l = nn_struct.size() - 1; l < 0; l--) {
            W[l] += -alpha * (1.0/m * tri_W[l]);
            b[l] += -alpha * (1.0/m * tri_b[l]);
        }
        cnt++;
    }
    return {W, b};
}


la::vector<double> predict(std::map<int, la::matrix<double>>& W, std::map<int, la::vector<double>>& b,
        std::vector<la::vector<double>>& x, int num_layers) {
    int m = x.size();
    auto y = generateVector(m, "zero");
    for (int i = 0; i < m; i++) {
        auto [h, z] = feed_forward(x[i], W, b);
        int max_ind = 0;
        double max_val = h[num_layers](0);
        for (int j = 1; j < h[num_layers].size(); j++) {
            if ( h[num_layers](j) > max_val) {
                max_val = h[num_layers](j);
                max_ind = j;
            }
        }
        y[i] = max_ind;
    }
    return y;
}