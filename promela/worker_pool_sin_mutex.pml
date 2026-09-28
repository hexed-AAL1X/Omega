#define NTREES 4
#define W 3
#define FIN 255

chan jobs = [NTREES] of { byte };

byte done = 0;
byte pendientes = W;

proctype Worker(byte id) {
	byte j;
	byte tmp;
	do
	:: jobs ? j ->
		if
		:: j == FIN -> break
		:: else ->
			tmp = done;
			tmp = tmp + 1;
			done = tmp
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
	assert(done == NTREES)
}
