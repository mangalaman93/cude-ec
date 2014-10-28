#ifndef INCLUDE_HASHFUNCTION_C_H
#define INCLUDE_HASHFUNCTION_C_H


#include <stdio.h>


//15
extern "C" unsigned int RSHash  (char* str, unsigned int len);
extern "C" unsigned int JSHash  (char* str, unsigned int len);
extern "C" unsigned int PJWHash (char* str, unsigned int len);
extern "C" unsigned int ELFHash (char* str, unsigned int len);
extern "C" unsigned int BKDRHash(char* str, unsigned int len);
extern "C" unsigned int SDBMHash(char* str, unsigned int len);
extern "C" unsigned int DJBHash (char* str, unsigned int len);
extern "C" unsigned int DEKHash (char* str, unsigned int len);
extern "C" unsigned int BPHash  (char* str, unsigned int len);
extern "C" unsigned int FNVHash (char* str, unsigned int len);
extern "C" unsigned int APHash  (char* str, unsigned int len);
extern "C" unsigned int krHash(char *str, unsigned int len);
extern "C" unsigned int ocaml_hash(char *str, unsigned int len) ;
extern "C" unsigned int sml_hash(char *str, unsigned int len);
extern "C" unsigned int stl_hash(char *str, unsigned int len);


#endif
