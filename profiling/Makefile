all:
	make vanilla.target
	make quantized.target

%.target: %.out
	bash setup_and_perf.sh $<

%.out: ../src/%.cpp
	g++ -Wall -Wno-unknown-pragmas --std=c++17 -O3 -I../src/bvh/include $< -o $@

clean:
	rm -rf vanilla.out quantized.out perf.data perf.data.old
	if [ -p ctl.fifo ]; then unlink ctl.fifo; fi
	if [ -p ack.fifo ]; then unlink ack.fifo; fi
