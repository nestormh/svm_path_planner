# Primera prueba algoritmo detección del suelo

load "Barrido.mat"

figure (1);
#polar( barrido1(:,1), barrido1(:,2) )
if (!exist("rotacion","var")) rotacion=0; endif
rotacion
l=barrido1 (:,2); alfa=(barrido1(:,1)-pi/2)+rotacion;
numPtos=length(l)
plot(l.*cos(alfa+pi/2),l.*sin(alfa+pi/2)) ;
grid on

Dmin=1; Dmax=3; Dinc=0.01;
TitaMax=rad(30);
TitaMin=-TitaMax;
TitaInc=rad(0.1);

rangoD=[Dmin:Dinc:Dmax];
numD=length(rangoD)
rangoTitas=[TitaMin:TitaInc:TitaMax];
numTitas=length(rangoTitas)

MatVota=zeros(length(rangoD),numTitas);

function tita=calculaTita(alfa,l,d)
	if(sin(alfa)==0.0) alfa, error("Seno de alfa se hace 0"); endif
	tita=atan((d/l-cos(alfa))/sin(alfa));
endfunction

usados=zeros(size(l)); #Para anotar los que son usados
usadosTeo=usados; #Para anotar los que son usados
numCalculos=0;
for indPto=[1:numPtos]
	alfaAct=alfa(indPto);
	lAct=l(indPto);
	if (abs(alfaAct)<rad(0.1))  #paracticamente 0
		indD=round((lAct-Dmin)/Dinc)+1;
		if (indD>0 && indD<=length(rangoD))
			#Vota a todos los titas de esa distancia
			MatVota(indD,:)+=ones(size(rangoTitas));
			usados(indPto)=1;
			usadosTeo(indPto)=1;
		endif
		continue
	endif
	usadosTeo(indPto)=1; #Teoricamente se usará
	#Probamos para ver si entan muy cerca
	tita=calculaTita(alfaAct,lAct,rangoD(1));
	if( ( (alfaAct>0) && (tita>TitaMax) ) 
		|| ( (alfaAct<0) && (tita<TitaMin) ) )
		#Esta fuera de rango
		usadosTeo(indPto)=0; #Teoricamente no se usará
		continue
	endif
	#Probamos si están muy lejos
	tita=calculaTita(alfaAct,lAct,rangoD(numD));
	if( ( (alfaAct>0) && (tita<TitaMin) ) 
		|| ( (alfaAct<0) && (tita>TitaMax) ) )
		#Esta fuera de rango
		usadosTeo(indPto)=0; #Teoricamente no se usará
		continue
	endif
	numCalculos++;
	for indd=[1:length(rangoD)]
		d=rangoD(indd);
		tita=calculaTita(alfaAct,lAct,d);
		indTita=round((tita-TitaMin)/TitaInc)+1;
		if(indTita>0 && indTita<=numTitas)
			MatVota(indd,indTita)++;
			usados(indPto)=1; #Se uso
			if ( usadosTeo(indPto)==0 )
				indPto
			endif
		endif
	endfor
endfor

figure (2);
#mesh(rangoTitas,rangoD,MatVota);

[maxD,indMD]=max(MatVota);
[maxTita,indTitaSel]=max(maxD);
maxTita
titaSel=rangoTitas(indTitaSel)
indDSel=indMD(indTitaSel);
dSel=rangoD(indDSel)
distintos=find(usadosTeo != usados);
numCalculos 
numUsados=length(find(usados))
numUsadosTeo=length(find(usadosTeo))
numDistintos=length(distintos)

#Encontramos los que maxTita que contribullen al máximo
DeltaD=1; DeltaTita=1;
contribuye=zeros(size(l));
for indPto=find(usados)'
	alfaAct=alfa(indPto);
	lAct=l(indPto);
	if (abs(alfaAct)<rad(0.1) )
		if ( abs((round((lAct-Dmin)/Dinc)+1)-indDSel)<=DeltaD )
			contribuye(indPto)=1;
		endif
		continue;
	endif
	for delta=[-DeltaD:DeltaD]
		tita=calculaTita(alfaAct,lAct,rangoD(indDSel+delta));
		indTita=round((tita-TitaMin)/TitaInc)+1;
		if(abs(indTita-indTitaSel)<=DeltaTita)
			contribuye(indPto)=1;
		endif
	endfor
endfor

numContribuyen=length(find(contribuye))

figure(1);
lr=4;
lUsados=l(find(usados)); alfaUsados=alfa(find(usados));
lContribuyen=l(find(contribuye)); alfaContribuyen=alfa(find(contribuye));
plot(l.*cos(alfa+pi/2),l.*sin(alfa+pi/2)
#  ,lUsados.*cos(alfaUsados+pi/2),lUsados.*sin(alfaUsados+pi/2),"x"
  ,lContribuyen.*cos(alfaContribuyen+pi/2),lContribuyen.*sin(alfaContribuyen+pi/2),"o"
 ,lr*[-cos(titaSel), cos(titaSel)],lr*[-sin(titaSel) ,sin(titaSel)]+dSel
  ,lr*[-cos(TitaMax), 0, cos(TitaMin)],lr*[-sin(TitaMax), 0 ,sin(TitaMin)]+Dmin
  ,lr*[-cos(TitaMin), 0, cos(TitaMax)],lr*[-sin(TitaMin), 0 ,sin(TitaMax)]+Dmax
) ;
grid on


