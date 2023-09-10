#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <queue>
#include <cassert>
#include <fstream>
#include <pthread.h>

std::multiset<std::string> x;
std::queue<std::string> y;
pthread_mutex_t x_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t y_mutex = PTHREAD_MUTEX_INITIALIZER;

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

    pthread_t threads[n];
    for (int i = 0; i < n; i++) {
        int errcode = pthread_create(&threads[i], NULL, producer, NULL);
        if (errcode) {
            std::runtime_error("Error creating thread, errcode = " + errcode);
        }
    }

    return 0;
}
