#!/bin/bash

for (( z=0; z<=30; z += 5 ))
do
	for (( s=0; s<4; s++ ))
	do
		for (( b1=1; b1<=9; b1 += 2 ))
		do
			for (( b2=1; b2<=9; b2 += 2 ))
			do

				./navbasadaentorno $s $z $b1 $b2
	
			done
		done
	done
done

