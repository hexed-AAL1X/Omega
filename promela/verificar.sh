#!/usr/bin/env bash
set -eu

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/resultados"
TMP="$(mktemp -d)"
mkdir -p "$OUT"
trap 'rm -rf "$TMP"' EXIT

compilar() {
	rm -f "$TMP"/pan*
	cp "$DIR/$1.pml" "$TMP/"
	(cd "$TMP" && spin -a "$1.pml" > /dev/null \
		&& gcc -O2 -DSAFETY -DNOCLAIM -o pan_seguridad pan.c \
		&& gcc -O2 -o pan pan.c)
}

verificar() {
	local modelo="$1" nombre="$2" binario="$3"
	shift 3
	echo "== $modelo :: $nombre ($binario $*)" | tee "$OUT/${modelo}_${nombre}.txt"
	(cd "$TMP" && "./$binario" "$@" || true) | tee -a "$OUT/${modelo}_${nombre}.txt" \
		| grep -E "errors:|assertion violated|acceptance cycle|invalid end states|States, stored" || true
	echo
}

compilar worker_pool
verificar worker_pool seguridad pan_seguridad
verificar worker_pool exclusion pan -a -N exclusion
verificar worker_pool termina pan -a -f -N termina

compilar pipeline
verificar pipeline seguridad pan_seguridad
verificar pipeline termina pan -a -f -N termina

compilar worker_pool_sin_mutex
verificar worker_pool_sin_mutex seguridad pan_seguridad
if [ -f "$TMP/worker_pool_sin_mutex.pml.trail" ]; then
	(cd "$TMP" && spin -t -p -g worker_pool_sin_mutex.pml) > "$OUT/worker_pool_sin_mutex_contraejemplo.txt"
	echo "Contraejemplo guardado en $OUT/worker_pool_sin_mutex_contraejemplo.txt"
fi
