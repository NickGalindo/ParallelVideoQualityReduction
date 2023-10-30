#!/bin/bash
bear -- make build video_reduction.cpp

printf "> DATA LOG FOR video_reduction <\n" > results.txt
printf "INPUT: %s\n" "$1" >> results.txt
printf "OUTPUT: %s\n\n" "$2" >> results.txt

for i in 32 16 8 4 2 1
do
  printf "> STARTING RUN FOR video_reduction ON %d THREADS <" "$i" >> results.txt
  { time ./video_reduction $1 $2 $i; } 2>> results.txt
  printf "> END OF RUN FOR video_reduction ON %d THREADS <\n\n" "$i" >> results.txt
done
