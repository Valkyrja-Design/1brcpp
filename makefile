run: compile
	./main
compile: main.cpp
	g++ --std=c++20 -o main main.cpp -O3 -flto -march=native -g -fno-omit-frame-pointer
test: compile
	./main testfiles/small.txt > output.txt
	diff output.txt testfiles/small_expected.txt
	rm output.txt
fine: compile
	hyperfine -w 2 './main ./measurements.txt'
record: compile
	perf record --call-graph dwarf -- ./main ./measurements.txt
clean:
	rm main
