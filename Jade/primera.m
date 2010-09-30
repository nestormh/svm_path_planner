# Primera prueba algoritmo detección del suelo

load "Barrido.mat"

figure (1);
#polar( barrido1(:,1), barrido1(:,2) )
l=barrido1 (:,2); alfa=(barrido1(:,1)-pi/2)+pi/6;
plot(l.*cos(alfa+pi/2),l.*sin(alfa+pi/2)) ;
grid on

Dmin=1; Dmax=3; Dinc=0.01;
TitaMin=rad(-45);
TitaMax=rad(45);
TitaInc=rad(0.1);

rangoD=[Dmin:Dinc:Dmax];
rangoTitas=[TitaMin:TitaInc:TitaMax];
numTitas=length(rangoTitas);

MatVota=zeros(length(rangoD),numTitas);


for ptac=[alfa l]'
	alfaAct=ptac(1);
	lAct=ptac(2);
	if (abs(alfaAct)<0.1)  #paracticamente 0
		indD=round((lAct-Dmin)/Dinc)+1;
		if (indD>0 && indD<=length(rangoD))
			#Vota a todos los titas de esa distancia
			MatVota(indD,:)+=ones(size(rangoTitas));
		endif
		continue
	endif
	for indd=[1:length(rangoD)]
		d=rangoD(indd);
#		tita=atan(sin(alfaAct)/(cos(alfaAct)-d/lAct));
		tita=atan((d/lAct-cos(alfaAct))/sin(alfaAct));
		indTita=round((tita-TitaMin)/TitaInc)+1;
		if(indTita>0 && indTita<=numTitas)
			MatVota(indd,indTita)++;
		endif
	endfor
endfor

figure (2);
#mesh(rangoTitas,rangoD,MatVota);

[maxD,indMD]=max(MatVota);
[maxTita,indMTita]=max(maxD);
titaSel=rangoTitas(indMTita)
dSel=rangoD(indMD(indMTita))

figure(1);
lr=4;
plot(l.*cos(alfa+pi/2),l.*sin(alfa+pi/2)
 ,lr*[-cos(titaSel), cos(titaSel)],lr*[-sin(titaSel) ,sin(titaSel)]+dSel
) ;
grid on


