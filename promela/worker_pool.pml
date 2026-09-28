#define NTREES 4
#define W 3
#define FIN 255

chan jobs = [NTREES] of { byte };

bit mutex = 0;
byte in_cs = 0;
byte done = 0;
byte pendientes = W;
bit trees[NTREES];
bool terminado = false;

inline lock() {
	atomic { mutex == 0 -> mutex = 1 }
}

inline unlock() {
	mutex = 0
}

proctype Worker(byte id) {
	byte j;
	do
	:: jobs ? j ->
		if
		:: j == FIN -> break
		:: else ->
			lock();
			in_cs++;
			assert(in_cs == 1);
			assert(trees[j] == 0);
			trees[j] = 1;
			done++;
			in_cs--;
			unlock()
		fi
	od;
	atomic { pendientes-- }
}

active proctype Main() {
	byte i;
	atomic {
		i = 0;
		do
		:: i < W -> run Worker(i); i++
		:: else -> break
		od
	};
	i = 0;
	do
	:: i < NTREES -> jobs ! i; i++
	:: else -> break
	od;
	i = 0;
	do
	:: i < W -> jobs ! FIN; i++
	:: else -> break
	od;
	pendientes == 0;
	assert(done == NTREES);
	i = 0;
	do
	:: i < NTREES -> assert(trees[i] == 1); i++
	:: else -> break
	od;
	terminado = true
}

ltl exclusion { [] (in_cs <= 1) }
ltl termina { <> terminado }
