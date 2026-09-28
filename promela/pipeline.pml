#define NBLOQUES 4
#define W 2
#define FIN 255

chan tareas = [NBLOQUES] of { byte };
chan parciales = [W] of { byte };

byte filas[NBLOQUES];
byte esperado = 0;
byte total = 0;
byte vivos = W;
bool cerrado = false;
bool terminado = false;

proctype Worker(byte id) {
	byte b;
	byte suma = 0;
	do
	:: tareas ? b ->
		if
		:: b == FIN -> break
		:: else -> suma = suma + filas[b]
		fi
	od;
	parciales ! suma;
	atomic { vivos-- }
}

proctype Cerrador() {
	vivos == 0;
	cerrado = true
}

active proctype Main() {
	byte i, p;
	atomic {
		i = 0;
		do
		:: i < NBLOQUES ->
			if
			:: filas[i] = 1
			:: filas[i] = 2
			:: filas[i] = 3
			fi;
			esperado = esperado + filas[i];
			i++
		:: else -> break
		od;
		i = 0;
		do
		:: i < W -> run Worker(i); i++
		:: else -> break
		od;
		run Cerrador()
	};
	i = 0;
	do
	:: i < NBLOQUES -> tareas ! i; i++
	:: else -> break
	od;
	i = 0;
	do
	:: i < W -> tareas ! FIN; i++
	:: else -> break
	od;
	do
	:: parciales ? p -> total = total + p
	:: cerrado && empty(parciales) -> break
	od;
	assert(total == esperado);
	terminado = true
}

ltl termina { <> terminado }
