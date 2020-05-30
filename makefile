
#default
default: all

all: clean tar

tar:
	tar -czvf ex2.tar multi_agents.py README.txt

clean:
	cd ..
	rm -rf *.tar

.PHONY: default tar clean