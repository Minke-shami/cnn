#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
using arry = std::array<float, 3>;
// 初始化权重的函数，添加注释解释其功能
float he_init(int in_dim) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, std::sqrt(2.0 / in_dim));
    return dis(gen);
}
// 生成数据集的函数，添加注释解释参数和功能
void create(std::vector<arry>& dataset, int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(0.0, 10.0);
    std::uniform_real_distribution<> dis_y(10.0, 20.0);
    while (num_samples--) {
        float x = dis_x(gen);
        float y = dis_y(gen);
        arry sample = {x, y, 0.0f};
        // 数据增强：随机平移，添加注释解释范围的选择
        float translation_x = std::uniform_real_distribution<>(-0.5, 0.5)(gen);
        float translation_y = std::uniform_real_distribution<>(-0.5, 0.5)(gen);
        x += translation_x;
        y += translation_y;
        if ((x >= 0.0 && x <= 5.0 && y >= 15.0 && y <= 20.0)) {
            sample[2] = 0.0f;
        } else if ((x > 5 && x <= 10.0 && y >= 10.0 && y < 15.0)) {
            sample[2] = 1.0f;
        } else if ((x >= 0.0 && x <= 5.0 && y >= 10.0 && y < 15.0)) {
            sample[2] = 2.0f;
        } else if ((x > 5.0 && x <= 10.0 && y >= 15.0 && y < 20.0)) {
            sample[2] = 3.0f;
        }
        dataset.push_back(sample);
    }
}
// 计算 softmax 的函数，添加注释解释算法步骤
std::vector<float> softmax(const std::vector<float>& inputs) {
    std::vector<float> result(inputs.size());
    float max_input = *std::max_element(inputs.begin(), inputs.end());
    float sum = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        result[i] = std::exp(inputs[i] - max_input);
        sum += result[i];
    }
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] /= sum;
    }
    return result;
}
// 激活函数类 Exponential ReLU，添加注释解释参数和功能
class ExponentialReLU {
public:
    ExponentialReLU() : alpha(1.0f) {}

    float operator()(float x) {
        return x > 0? x : alpha * (std::exp(x) - 1);
    }

    float derivative(float x) {
        return x > 0? 1.0f : (*this)(x) + alpha;
    }

private:
    float alpha;
};
// 数据归一化函数，添加注释解释算法步骤
void normalize_data(std::vector<arry>& data) {
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto& sample : data) {
        min_x = std::min(min_x, sample[0]);
        max_x = std::max(max_x, sample[0]);
        min_y = std::min(min_y, sample[1]);
        max_y = std::max(max_y, sample[1]);
    }

    for (auto& sample : data) {
        sample[0] = (sample[0] - min_x) / (max_x - min_x);
        sample[1] = (sample[1] - min_y) / (max_y - min_y);
    }
}
// 模拟学习率调度器的结构体，添加注释解释成员变量的作用
struct LearningRateScheduler {
    float learningRate;
    float minLearningRate;
    int decaySteps;
    LearningRateScheduler(float initialRate, float minRate, int steps)
        : learningRate(initialRate), minLearningRate(minRate), decaySteps(steps) {}

    float get_lr() {
        return learningRate;
    }
};
// 神经网络类，添加注释解释成员变量和函数的作用
class simpleNN {
public:
    // 构造函数，添加注释解释参数的作用
    simpleNN(int input_size, int hidden_size1, int hidden_size2, int output_size)
            : input_size(input_size), hidden_size1(hidden_size1), hidden_size2(hidden_size2), output_size(output_size),
              learning_rate(0.0001), beta1(0.9), beta2(0.999), epsilon(1e-8) {
        weights_input_hidden1.resize(input_size * hidden_size1);
        for (auto& w : weights_input_hidden1) w = he_init(input_size);
        weights_hidden1_hidden2.resize(hidden_size1 * hidden_size2);
        for (auto& w : weights_hidden1_hidden2) w = he_init(hidden_size1);
        weights_hidden2_output.resize(hidden_size2 * output_size);
        for (auto& w : weights_hidden2_output) w = he_init(hidden_size2);
        m_input_hidden1.resize(weights_input_hidden1.size(), 0.0f);
        v_input_hidden1.resize(weights_input_hidden1.size(), 0.0f);
        m_hidden1_hidden2.resize(weights_hidden1_hidden2.size(), 0.0f);
        v_hidden1_hidden2.resize(weights_hidden1_hidden2.size(), 0.0f);
        m_hidden2_output.resize(weights_hidden2_output.size(), 0.0f);
        v_hidden2_output.resize(weights_hidden2_output.size(), 0.0f);
        // 初始化学习率调度器
         scheduler = std::make_unique<LearningRateScheduler>(learning_rate, 0.00001, 1500);
    }
    // 前向传播函数，添加注释解释算法步骤
    void forward(const std::vector<float>& inputs) {
        hidden_layer1.clear();
        ExponentialReLU elu;
        for (int i = 0; i < hidden_size1; ++i) {
            float activation = 0.0;
            for (int j = 0; j < input_size; ++j) activation += inputs[j] * weights_input_hidden1[j + i * input_size];
            hidden_layer1.push_back(elu(activation));
        }

        hidden_layer2.clear();
        ExponentialReLU elu2;
        for (int i = 0; i < hidden_size2; ++i) {
            float activation = 0.0;
            for (int j = 0; j < hidden_size1; ++j) activation += hidden_layer1[j] * weights_hidden1_hidden2[j + i * hidden_size1];
            hidden_layer2.push_back(elu2(activation));
        }

        output_layer.clear();
        for (int i = 0; i < output_size; ++i) {
            float activation = 0.0;
            for (int j = 0; j < hidden_size2; ++j) activation += hidden_layer2[j] * weights_hidden2_output[j + i * hidden_size2];
            output_layer.push_back(activation);
        }

        output_layer = softmax(output_layer);
    }
    // 反向传播函数，添加注释解释算法步骤
    void backward(const std::vector<float>& inputs, const std::vector<float>& targets) {
        ExponentialReLU elu;
        ExponentialReLU elu2;
        std::vector<float> out_errors(output_size);
        for (int i = 0; i < output_size; ++i) out_errors[i] = targets[i] - output_layer[i];

        // 梯度裁剪
        const float clip_value = 5.0f;
        for (auto& error : out_errors) {
            if (error > clip_value) error = clip_value;
            if (error < -clip_value) error = -clip_value;
        }
        for (int i = 0; i < hidden_size2; ++i) {
            for (int j = 0; j < output_size; ++j) {
                float delta = learning_rate * out_errors[j] * hidden_layer2[i];
                update_weights(weights_hidden2_output, m_hidden2_output, v_hidden2_output, i + j * hidden_size2, delta);
            }
        }
        std::vector<float> hidden2_errors(hidden_size2, 0.0f);
        for (int i = 0; i < hidden_size2; ++i) {
            for (int j = 0; j < output_size; ++j) hidden2_errors[i] += out_errors[j] * weights_hidden2_output[i + j * hidden_size2];
            hidden2_errors[i] *= elu2.derivative(hidden_layer2[i]);
        }
        for (int i = 0; i < hidden_size1; ++i) {
            for (int j = 0; j < hidden_size2; ++j) {
                float delta = learning_rate * hidden2_errors[j] * hidden_layer1[i];
                update_weights(weights_hidden1_hidden2, m_hidden1_hidden2, v_hidden1_hidden2, i + j * hidden_size1, delta);
            }
        }
        std::vector<float> hidden1_errors(hidden_size1, 0.0f);
        for (int i = 0; i < hidden_size1; ++i) {
            for (int j = 0; j < hidden_size2; ++j) hidden1_errors[i] += hidden2_errors[j] * weights_hidden1_hidden2[i + j * hidden_size1];
            hidden1_errors[i] *= elu.derivative(hidden_layer1[i]);
        }
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size1; ++j) {
                float delta = learning_rate * hidden1_errors[j] * inputs[i];
                update_weights(weights_input_hidden1, m_input_hidden1, v_input_hidden1, i + j * input_size, delta);
            }
        }
    }
    // 训练函数，添加注释解释算法步骤和参数的作用
    void train(const std::vector<arry>& data, int epochs, int batch_size) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            for (int batch_start = 0; batch_start < data.size(); batch_start += batch_size) {
                int batch_end = std::min(batch_start + batch_size, static_cast<int>(data.size()));
                for (int i = batch_start; i < batch_end; ++i) {
                    const auto& sample = data[i];
                    std::vector<float> inputs = {sample[0], sample[1]};
                    std::vector<float> targets(output_size, 0.0f);
                    targets[static_cast<int>(sample[2])] = 1.0f;
                    forward(inputs);
                    total_loss += compute_loss(targets);
                    backward(inputs, targets);
                }
            }
            // 更新学习率
            learning_rate = scheduler->get_lr();
            if (epoch % 100 == 0) std::cout << "Epoch " << epoch << ", Loss: " << total_loss / data.size() << std::endl;
            if (epoch % 200 == 0 && epoch!= 0) learning_rate *= 0.9;
            if (epoch % 500 == 0 && epoch!= 0) learning_rate *= 0.75;
        }
        
    }
    // 计算损失函数，添加注释解释算法步骤
    float compute_loss(const std::vector<float>& targets) {
        float loss = 0.0f;
        for (int i = 0; i < output_size; ++i)
            loss -= targets[i] * std::log(output_layer[i] + 1e-10);
        return loss;
    }
    // 测试函数，添加注释解释算法步骤和返回值的意义
    float test(const std::vector<arry>& data) {
        int correct_prediction = 0;
        for (const auto& sample : data) {
            std::vector<float> inputs = {sample[0], sample[1]};
            forward(inputs);
            int predicted_label = static_cast<int>(std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end())));
            if (predicted_label == static_cast<int>(sample[2])) ++correct_prediction;
        }
        return static_cast<float>(correct_prediction) / data.size();
    }
private:
    int input_size;
    int hidden_size1;
    int hidden_size2;
    int output_size;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    std::vector<float> weights_input_hidden1;
    std::vector<float> weights_hidden1_hidden2;
    std::vector<float> weights_hidden2_output;
    std::vector<float> hidden_layer1;
    std::vector<float> hidden_layer2;
    std::vector<float> output_layer;
    std::vector<float> m_input_hidden1;
    std::vector<float> v_input_hidden1;
    std::vector<float> m_hidden1_hidden2;
    std::vector<float> v_hidden1_hidden2;
    std::vector<float> m_hidden2_output;
    std::vector<float> v_hidden2_output;
    // 学习率调度器
    std::unique_ptr<LearningRateScheduler> scheduler;

    // 更新权重的函数，添加注释解释算法步骤和参数的作用
    void update_weights(std::vector<float>& weights, std::vector<float>& m, std::vector<float>& v, int index, float delta) {
        m[index] = beta1 * m[index] + (1 - beta1) * delta;
        v[index] = beta2 * v[index] + (1 - beta2) * delta * delta;
        float m_hat = m[index] / (1 - beta1);
        float v_hat = v[index] / (1 - beta2);
        weights[index] += learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
};

int main() {
    try {
        // 生成训练集和测试集，这里假设生成数据集过程中不会出错，但实际应用中可以添加错误处理
        std::vector<arry> train;
        std::vector<arry> test;
        create(train, 5000);
        create(test, 500);
        // 数据归一化
        normalize_data(train);
        normalize_data(test);

        // 创建神经网络对象，可以通过参数传递的方式使代码更具通用性
        simpleNN nn(2, 40, 20, 4);

        for (int i = 0; i < 10; ++i) {
            // 训练模型，打印训练过程中的损失和准确率
            nn.train(train, 1000, 32);
            float accuracy = nn.test(test);
            std::cout << "Accuracy after training iteration " << i << ": " << accuracy * 100.0 << "%" << std::endl;
        }
    } catch (const std::exception& e) {
        // 捕获并打印可能出现的异常
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }
    return 0;
}
