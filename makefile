
#default
default: all

all: clean tar

tar:
	tar -czvf ex2.tar search.py blokus_problems.py README.txt

clean:
	cd ..
	rm -rf *.tar

.PHONY: default tar clean