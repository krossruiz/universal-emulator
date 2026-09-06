/*
 * GBA emulator - milestone 1: ROM header reader.
 * x86-64 assembly (GAS, Intel syntax), linked against the mingw-w64 C runtime
 * so we can call fopen/fread/printf directly using the Microsoft x64 calling
 * convention (first four args in RCX, RDX, R8, R9; 32 bytes of shadow space
 * must be reserved by the caller before any call).
 */
    .intel_syntax noprefix
    .text

    .extern fopen
    .extern fread
    .extern fclose
    .extern printf
    .extern exit

    .global main

/* ---- rodata ---- */
    .section .rodata
mode_rb:        .asciz "rb"
default_path:   .asciz "..\\Legend of Zelda, The - A Link to the Past & Four Swords (USA).gba"

msg_usage:      .asciz "usage: gba_emu.exe [rom.gba]  (no path given, using default)\n"
msg_open_fail:  .asciz "error: could not open ROM file\n"
msg_read_fail:  .asciz "error: could not read ROM header (file too short?)\n"
msg_header:     .asciz "== GBA ROM header ==\n"
msg_entry:      .asciz "entry point bytes : %02X %02X %02X %02X\n"
msg_title:      .asciz "title              : %.12s\n"
msg_gamecode:   .asciz "game code          : %.4s\n"
msg_makercode:  .asciz "maker code         : %.2s\n"
msg_fixed_ok:   .asciz "fixed value (0x96) : OK\n"
msg_fixed_bad:  .asciz "fixed value (0x96) : MISMATCH (0x%02X) - not a valid GBA ROM?\n"
msg_romsize:    .asciz "bytes read         : %d\n"

/* ---- bss ---- */
    .section .bss
    .align 16
header_buf:     .skip 256       /* GBA cartridge header is 0xC0 (192) bytes */

/* ---- text ---- */
    .text
main:
    push rbp
    mov  rbp, rsp
    sub  rsp, 96            /* shadow space (32) + locals, 16-byte aligned */

    /* locals:
     *   [rbp-8]  = argc (int, passed in ecx)
     *   [rbp-16] = argv (char**, passed in rdx)
     *   [rbp-24] = FILE* fp
     */
    mov  [rbp-8], ecx
    mov  [rbp-16], rdx

    /* pick ROM path: argv[1] if argc > 1, else default_path */
    mov  eax, [rbp-8]
    cmp  eax, 2
    jl   use_default

    mov  rax, [rbp-16]      /* argv */
    mov  rax, [rax+8]       /* argv[1] */
    jmp  have_path

use_default:
    lea  rcx, [rip+msg_usage]
    call printf
    lea  rax, [rip+default_path]

have_path:
    mov  r12, rax            /* r12 = path pointer (callee-saved) */

    /* FILE* fp = fopen(path, "rb"); */
    mov  rcx, r12
    lea  rdx, [rip+mode_rb]
    call fopen
    test rax, rax
    jz   open_failed
    mov  [rbp-24], rax

    /* size_t n = fread(header_buf, 1, 192, fp); */
    lea  rcx, [rip+header_buf]
    mov  edx, 1
    mov  r8d, 192
    mov  r9, [rbp-24]
    call fread
    mov  r13d, eax            /* r13d = bytes read */

    mov  rcx, [rbp-24]
    call fclose

    cmp  r13d, 192
    jl   read_failed

    lea  rcx, [rip+msg_header]
    call printf

    /* entry point (first 4 bytes, little-endian display) */
    lea  rcx, [rip+msg_entry]
    movzx edx, byte ptr [rip+header_buf+0]
    movzx r8d, byte ptr [rip+header_buf+1]
    movzx r9d, byte ptr [rip+header_buf+2]
    sub  rsp, 32
    movzx eax, byte ptr [rip+header_buf+3]
    mov  [rsp+32], eax
    call printf
    add  rsp, 32

    /* title: 12 bytes at offset 0xA0 */
    lea  rcx, [rip+msg_title]
    lea  rdx, [rip+header_buf+0xA0]
    call printf

    /* game code: 4 bytes at offset 0xAC */
    lea  rcx, [rip+msg_gamecode]
    lea  rdx, [rip+header_buf+0xAC]
    call printf

    /* maker code: 2 bytes at offset 0xB0 */
    lea  rcx, [rip+msg_makercode]
    lea  rdx, [rip+header_buf+0xB0]
    call printf

    /* fixed value at 0xB2 must be 0x96 */
    movzx eax, byte ptr [rip+header_buf+0xB2]
    cmp  eax, 0x96
    jne  fixed_bad
    lea  rcx, [rip+msg_fixed_ok]
    call printf
    jmp  fixed_done
fixed_bad:
    lea  rcx, [rip+msg_fixed_bad]
    mov  edx, eax
    call printf
fixed_done:

    lea  rcx, [rip+msg_romsize]
    mov  edx, r13d
    call printf

    xor  ecx, ecx
    jmp  done

open_failed:
    lea  rcx, [rip+msg_open_fail]
    call printf
    mov  ecx, 1
    jmp  done

read_failed:
    lea  rcx, [rip+msg_read_fail]
    call printf
    mov  ecx, 1
    jmp  done

done:
    mov  eax, ecx
    leave
    ret
