all:
	make vanilla.target
	make quantized.target

%.target: %.out
	if [ -z ${SCENE} ]; then false; fi
	bash setup_and_perf.sh $< ${SCENE}

%.out: %.cpp
	g++ -Wall -Wno-unknown-pragmas --std=c++17 -O3 -I../src/bvh/include -DNDEBUG $< -o $@

clean:
	rm -rf vanilla.out quantized.out perf.data perf.data.old
	if [ -p ctl.fifo ]; then unlink ctl.fifo; fi
	if [ -p ack.fifo ]; then unlink ack.fifo; fi
