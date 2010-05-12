#!/bin/bash
firstTime=0

for (( testIdx=0; testIdx<1; testIdx += 1))
do	
	if [ $testIdx -eq 0 ]; then
		rm /home/neztol/doctorado/Datos/EstadisticasITER/ram/*
		cp /home/neztol/doctorado/Datos/EstadisticasITER/tripode1/* /home/neztol/doctorado/Datos/EstadisticasITER/ram
	fi
	if [ $testIdx -eq 1 ]; then
		rm /home/neztol/doctorado/Datos/EstadisticasITER/ram/*
		cp /home/neztol/doctorado/Datos/EstadisticasITER/tripode2/* /home/neztol/doctorado/Datos/EstadisticasITER/ram
	fi
	if [ $testIdx -eq 2 ]; then
		rm /home/neztol/doctorado/Datos/EstadisticasITER/ram/*
		cp /home/neztol/doctorado/Datos/EstadisticasITER/tripode3/* /home/neztol/doctorado/Datos/EstadisticasITER/ram
	fi	
#	for (( idx=0; idx<=55; idx += 1 ))
	for (( idx=0; idx<27; idx += 1 ))
	do
		if [ $firstTime -eq 0 ]; then
		  idx=26
		fi
		for (( z=0; z<=30; z += 5 ))
		do
			if [ $firstTime -eq 0 ]; then
				z=0
			fi
			for (( s=0; s<3; s++ ))
			do
				if [ $firstTime -eq 0 ]; then
					s=0
				fi
				for (( b1=1; b1<=9; b1 += 2 ))
				do
					if [ $firstTime -eq 0 ]; then
						b1=1
					fi
					for (( b2=1; b2<=9; b2 += 2 ))
					do
						if [ $firstTime -eq 0 ]; then
							b2=1
							firstTime=1;  
						fi
#						echo $testIdx $idx $s $z $b1 $b2						
						./navbasadaentorno $testIdx $idx $s $z $b1 $b2
					done
				done
			done
		done
	done	
done

