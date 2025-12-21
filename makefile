run: compile
	./main
compile: main.cpp
	g++ --std=c++20 -o main main.cpp -O3
test: compile
	./main testfiles/small.txt > output.txt
	diff output.txt testfiles/small_expected.txt
	./main testfiles/big.txt > output.txt
	diff output.txt testfiles/big_expected.txt
	rm output.txt
clean:
	rm main
