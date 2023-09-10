#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <queue>
#include <cassert>
#include <fstream>
#include <sstream>
#include <pthread.h>

std::multiset<std::string> x;
std::queue<std::string> y;
std::map<std::string, int> z;
pthread_mutex_t x_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t y_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t z_mutex = PTHREAD_MUTEX_INITIALIZER;

void *producer(void *arg) {
    pthread_mutex_lock(&x_mutex);
    std::string file = *x.begin();
    x.erase(x.begin());
    pthread_mutex_unlock(&x_mutex);

    std::ifstream ifs(file);
    std::string line;

    while (getline(ifs, line)) {
        pthread_mutex_lock(&y_mutex);
        y.push(line);
        pthread_mutex_unlock(&y_mutex);
    }

    return NULL;
}

void *consumer(void *arg) {
    pthread_mutex_lock(&y_mutex);
    std::string line = y.front();
    y.pop();
    pthread_mutex_unlock(&y_mutex);

    std::stringstream ss(line);
    std::string word;
    while (ss >> word) {
        pthread_mutex_lock(&z_mutex);
        z[word]++;
        pthread_mutex_unlock(&z_mutex);
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::string input(argv[1]);

    std::ifstream ifs(input);
    std::string file;
    while (ifs >> file) {
        x.insert(file);
    }

    int n = x.size();

    pthread_t producerThreads[n];
    for (int i = 0; i < n; i++) {
        int errcode = pthread_create(&producerThreads[i], NULL, producer, NULL);
        if (errcode) {
            std::runtime_error("Error creating thread, errcode = " + errcode);
        }
    }

    int m = n;
    pthread_t consumerThreads[m];
    for (int i = 0; i < m; i++) {
        int errcode = pthread_create(&consumerThreads[i], NULL, consumer, NULL);
        if (errcode) {
            std::runtime_error("Error creating thread, errcode = " + errcode);
        }
    }

    for (auto &word : z) {
        std::cout << word.first << " " << word.second << std::endl;
    }

    return 0;
}
