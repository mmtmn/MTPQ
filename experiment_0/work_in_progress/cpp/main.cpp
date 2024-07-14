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
#include <condition_variable>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace sf;
using namespace Eigen;
using namespace tiny_dnn;
using namespace tiny_dnn::layers;

constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 600;
const sf::Color BACKGROUND_COLOR = sf::Color(255, 255, 255);
const sf::Color AGENT_COLOR = sf::Color(0, 0, 255);
const sf::Color FOOD_COLOR = sf::Color(0, 255, 0);
constexpr int AGENT_SIZE = 10;
constexpr int FOOD_SIZE = 5;

mutex mtx;
condition_variable cv;
priority_queue<pair<int, function<void()>>> taskQueue;

struct Food {
    sf::Vector2f position;

    Food(float x, float y) : position(x, y) {}

    void draw(RenderWindow& window) {
        CircleShape shape(FOOD_SIZE);
        shape.setFillColor(FOOD_COLOR);
        shape.setPosition(position.x, position.y);
        window.draw(shape);
    }
};

struct Task {
    string name;
    int priority;
    double complexity;
    function<void()> action;

    Task(string n, int p, double c, function<void()> a) : name(move(n)), priority(p), complexity(c), action(move(a)) {}

    bool operator<(const Task& other) const {
        return priority > other.priority; // Higher priority first
    }
};

Vector4f getRealTimeData();
float predictPriority(network<sequential>& net, const Vector4f& data);

class Agent {
public:
    string name;
    sf::Vector2f position;
    sf::Vector2f direction;
    priority_queue<Task> tasks;
    atomic<float> energy;
    bool alive = true;
    float stress = 0;

    Agent(string n, float x, float y, float e) : name(std::move(n)), position(x, y), direction(1, 0), energy(e) {}

    Agent(Agent&& other) noexcept 
        : name(std::move(other.name)), position(other.position), direction(other.direction), tasks(std::move(other.tasks)), 
          energy(other.energy.load()), alive(other.alive), stress(other.stress) {}

    Agent& operator=(Agent&& other) noexcept {
        if (this != &other) {
            name = std::move(other.name);
            position = other.position;
            direction = other.direction;
            tasks = std::move(other.tasks);
            energy.store(other.energy.load());
            alive = other.alive;
            stress = other.stress;
        }
        return *this;
    }

    void addTask(const Task& task) {
        lock_guard<mutex> lock(mtx);
        tasks.push(task);
        cv.notify_all();
    }

    void processTasks() {
        while (alive) {
            unique_lock<mutex> lock(mtx);
            cv.wait(lock, [this] { return !tasks.empty() || !alive; });

            while (!tasks.empty() && alive) {
                Task task = tasks.top();
                tasks.pop();
                lock.unlock();
                if (energy >= task.complexity) {
                    task.action();
                    float currentEnergy = energy.load();
                    energy.store(currentEnergy - task.complexity);
                }
                lock.lock();
            }
        }
    }

    void simulateLife() {
        thread taskProcessor(&Agent::processTasks, this);
        taskProcessor.detach();
    }

    void draw(RenderWindow& window) {
        CircleShape shape(AGENT_SIZE);
        shape.setFillColor(AGENT_COLOR);
        shape.setPosition(position.x, position.y);
        window.draw(shape);
    }

    void moveAgent() {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(-1, 1);
        direction = sf::Vector2f(dist(gen), dist(gen));
        position += direction;
        position = sf::Vector2f(max(0.f, min(position.x, float(SCREEN_WIDTH))),
                            max(0.f, min(position.y, float(SCREEN_HEIGHT))));
        cout << name << " moved to " << position.x << ", " << position.y << endl;
    }

    void stayPut() {
        cout << name << " is staying put." << endl;
    }

    void reproduce(vector<Agent>& agents) {
        if (energy > 50) {
            agents.emplace_back(name + "_child", position.x, position.y, 50);
            float currentEnergy = energy.load();
            energy.store(currentEnergy - 50);
            cout << name << " has reproduced." << endl;
        }
    }

    void senseEnvironment(const vector<Food>& foods, const vector<Agent>& agents) {
        vision(foods);
        hearing(agents);
        touch(agents);
    }

    void vision(const vector<Food>& foods) {
        float visionRange = 50;
        float visionAngle = M_PI / 4;
        for (const auto& food : foods) {
            sf::Vector2f toFood = food.position - position;
            float distance = sqrt(toFood.x * toFood.x + toFood.y * toFood.y);
            if (distance < visionRange) {
                float angle = acos((direction.x * toFood.x + direction.y * toFood.y) /
                                   (sqrt(direction.x * direction.x + direction.y * direction.y) * distance));
                if (angle < visionAngle) {
                    float currentEnergy = energy.load();
                    energy.store(currentEnergy + 10);
                    cout << name << " found food by vision and increased energy to " << energy << endl;
                    break;
                }
            }
        }
    }

    void hearing(const vector<Agent>& agents) {
        float hearingRange = 100;
        for (const auto& agent : agents) {
            if (this != &agent) {
                float distance = sqrt((position.x - agent.position.x) * (position.x - agent.position.x) +
                                      (position.y - agent.position.y) * (position.y - agent.position.y));
                if (distance < hearingRange) {
                    cout << name << " hears " << agent.name << endl;
                }
            }
        }
    }

    void touch(const vector<Agent>& agents) {
        for (const auto& agent : agents) {
            if (this != &agent) {
                float distance = sqrt((position.x - agent.position.x) * (position.x - agent.position.x) +
                                      (position.y - agent.position.y) * (position.y - agent.position.y));
                if (distance < AGENT_SIZE) {
                    cout << name << " touches " << agent.name << endl;
                    stress += 10;
                }
            }
        }
    }

    void decideNextAction(vector<Agent>& agents, network<sequential>& net) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.1, 0.5);

        vector<function<void()>> actions = {bind(&Agent::moveAgent, this), bind(&Agent::stayPut, this)};
        if (energy > 50) {
            actions.push_back(bind(&Agent::reproduce, this, ref(agents)));
        }

        for (const auto& action : actions) {
            float priority = predictPriority(net, getRealTimeData()) - float(stress);
            addTask(Task("Action", priority, dis(gen), action));
        }
    }

    void update(const vector<Food>& foods, vector<Agent>& agents, network<sequential>& net) {
        if (alive) {
            senseEnvironment(foods, agents);
            decideNextAction(agents, net);
            processTasks();
            if (energy <= 0) {
                alive = false;
                cout << name << " has died." << endl;
            }
        }
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

vector<Agent> agents;
vector<Food> foods;
network<sequential> net;

void initializeSimulation() {
    net = setupNeuralNetwork();
    agents.emplace_back("Agent1", 400, 300, 100);
    foods.emplace_back(100, 100);

    for (auto& agent : agents) {
        agent.simulateLife();
    }
}

void runSimulation() {
    RenderWindow window(VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Agent-Based Simulation");
    window.setFramerateLimit(60);

    initializeSimulation();

    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed)
                window.close();
        }

        window.clear(BACKGROUND_COLOR);

        for (auto& food : foods) {
            food.draw(window);
        }

        for (auto& agent : agents) {
            agent.update(foods, agents, net);
            agent.draw(window);
        }

        window.display();
    }
}

int main() {
    runSimulation();
    return 0;
}
