all: pgm.o hough

hough: hough.cu pgm.o
	@echo "Compiling the hough transform program..."
	nvcc hough.cu pgm.o -o hough `pkg-config --cflags --libs opencv4`

pgm.o: common/pgm.cpp
	@echo "Compiling the pgm object..."
	g++ -c common/pgm.cpp -o pgm.o
