CXX      = g++
CXXFLAGS = -std=gnu++17 -O3 -DNDEBUG -march=native -funroll-loops
TARGET   = our_hparter

SRCS = partition.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
