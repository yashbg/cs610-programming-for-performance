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

void *producer(void *arg) {
    
}

int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::string input(argv[1]);

    std::ifstream ifs(input);
    std::string line;
    while (ifs >> line) {
        x.insert(line);
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
