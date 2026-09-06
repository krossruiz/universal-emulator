#!/usr/bin/env bash
# Assembles and links the GBA emulator (x86-64 assembly, mingw-w64 toolchain).
set -e
cd "$(dirname "$0")"
gcc -c src/main.S -o build/main.o
gcc build/main.o -o build/gba_emu.exe
echo "built build/gba_emu.exe"
