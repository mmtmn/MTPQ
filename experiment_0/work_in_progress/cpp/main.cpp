#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include <tiny_dnn/tiny_dnn.h>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <iostream>
#include <random>
#include <functional>
#include <atomic>
#include <chrono>

using namespace std;
using namespace Eigen;
using sfColor = sf::Color; // Specific alias for SFML color to avoid confusion with other libraries
using namespace tiny_dnn;
using namespace tiny_dnn::layers;

constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 600;
const sfColor BACKGROUND_COLOR = sfColor(255, 255, 255);
const sfColor AGENT_COLOR = sfColor(0, 0, 255);
const sfColor FOOD_COLOR = sfColor(0, 255, 0);
constexpr int AGENT_SIZE = 10;
constexpr int FOOD_SIZE = 5;

mutex mtx;
priority_queue<pair<int, function<void()>>> taskQueue;

struct Task {
    string name;
    int priority;
    double complexity;
    function<void()> action;

    Task(string n, int p, double c, function<void()> a) : name(move(n)), priority(p), complexity(c), action(move(a)) {}

    bool operator<(const Task& other) const {
        return priority < other.priority;
    }
};

class Agent {
public:
    string name;
    Vector2f position;
    priority_queue<Task> tasks;
    atomic<float> energy;
    bool alive = true;

    Agent(string n, float x, float y, float e) : name(move(n)), position(x, y), energy(e) {}

    // Custom move constructor
    Agent(Agent&& other) noexcept 
        : name(move(other.name)), position(move(other.position)), tasks(move(other.tasks)), 
          energy(other.energy.load()), alive(other.alive) {}

    // Custom move assignment operator
    Agent& operator=(Agent&& other) noexcept {
        if (this != &other) {
            name = move(other.name);
            position = move(other.position);
            tasks = move(other.tasks);
            energy.store(other.energy.load());
            alive = other.alive;
        }
        return *this;
    }

    void processTasks() {
        while (!tasks.empty() && energy > 0) {
            Task task = tasks.top();
            tasks.pop();
            float currentEnergy = energy.load();
            if (currentEnergy >= task.complexity) {
                task.action();
                currentEnergy -= task.complexity;
                energy.store(currentEnergy);
            }
        }
        if (energy <= 0) alive = false;
    }

    void draw(sf::RenderWindow& window) {
        sf::CircleShape shape(AGENT_SIZE);
        shape.setFillColor(AGENT_COLOR);
        shape.setPosition(position.x(), position.y());
        window.draw(shape);
    }
};

network<sequential> setupNeuralNetwork() {
    network<sequential> net;
    net << fully_connected_layer(4, 20) << tanh_layer()
        << fully_connected_layer(20, 1) << sigmoid_layer();
    return net;
}

Vector4f getRealTimeData() {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d_light(500, 50), d_sound(300, 50), d_touch(150, 30);
    uniform_real_distribution<> stress(0, 1);
    Vector4f data(d_light(gen), d_sound(gen), d_touch(gen), stress(gen));
    return data;
}

float predictPriority(network<sequential>& net, const Vector4f& data) {
    vec_t input = {data[0], data[1], data[2], data[3]};
    return net.predict(input)[0];
}

void runSimulation() {
    sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Agent-Based Simulation with MTPQ");
    window.setFramerateLimit(60);

    vector<Agent> agents;
    agents.emplace_back("Agent1", 400, 300, 100);

    network<sequential> net = setupNeuralNetwork();

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(BACKGROUND_COLOR);

        for (auto& agent : agents) {
            if (agent.alive) {
                agent.processTasks();
                agent.draw(window);
            }
        }

        window.display();
    }
}

int main() {
    runSimulation();
    return 0;
}
