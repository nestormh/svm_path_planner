# Pasamos a apuntar los que contribuyen a cada votación.

load "Barrido.mat"
figure (1);
#polar( barrido1(:,1), barrido1(:,2) )
if (!exist("rotacion","var")) rotacion=0; endif
rotacion
l=barrido1 (:,2); alfa=(barrido1(:,1)-pi/2)+rotacion;
numPtos=length(l)
plot(l.*cos(alfa+pi/2),l.*sin(alfa+pi/2)) ;
grid on

Dmin=1; Dmax=3; 
if (!exist("Dinc","var")) Dinc=0.01; endif
Dinc

TitaMax=rad(30); TitaMin=-TitaMax;
if (!exist("TitaInc","var")) TitaInc=rad(0.1); endif
incGrados=degree(TitaInc)

rangoD=[Dmin:Dinc:Dmax];
numD=length(rangoD)
rangoTitas=[TitaMin:TitaInc:TitaMax];
numTitas=length(rangoTitas)

#Ahora cada fila de MatVotaQuien apuntará los puntos que votan cada opción
#  El primer elemento indica cuantos hay usados en la fila.
MatVotaQuien=zeros(numD*numTitas,numPtos+1);

function tita=calculaTita(alfa,l,d)
	if(sin(alfa)==0.0) alfa, error("Seno de alfa se hace 0"); endif
	tita=atan((d/l-cos(alfa))/sin(alfa));
endfunction


function ApuntaEnMatVotaQuien(id,iT,iP,MVQ,nT)
	indMV=(id-1)*nT+iT;
	[ indMV MVQ(indMV,[1:10])]
	posAct=++MVQ(indMV,1);
	MVQ(indMV,posAct+1)=iP;
#	keyboard
	[ indMV MVQ(indMV,[1:10])]
endfunction

usados=zeros(size(l)); #Para anotar los que son usados
usadosTeo=usados; #Para anotar los que son usados
numCalculos=0;
Traza=zeros(numD*numPtos,10); indTraza=1;
teoricosPeroNoUsados=[]; usadosPeroNoTeoricos=[];
for indPto=[1:numPtos]
	alfaAct=alfa(indPto);
	lAct=l(indPto);
	if (abs(alfaAct)<rad(0.1))  #paracticamente 0
		indD=round((lAct-Dmin)/Dinc)+1;
		if (indD>0 && indD<=length(rangoD))
			#Vota a todos los titas de esa distancia
			for i=1:numTitas
				#ApuntaEnMatVotaQuien(indD,i,indPto,MatVotaQuien,numTitas);
				indMV=(indD-1)*numTitas+i;
				posAct=++MatVotaQuien(indMV,1);
				MatVotaQuien(indMV,posAct+1)=indPto;
			endfor
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
		#continue
	endif
	#Probamos si están muy lejos
	tita=calculaTita(alfaAct,lAct,rangoD(numD));
	if( ( (alfaAct>0) && (tita<TitaMin) ) 
		|| ( (alfaAct<0) && (tita>TitaMax) ) )
		#Esta fuera de rango
		usadosTeo(indPto)=0; #Teoricamente no se usará
		#continue
	endif
	numCalculos++;
	d=Dmin-Dinc/2;
	tita2=calculaTita(alfaAct,lAct,d);
	indTita2=round((tita2-TitaMin)/TitaInc)+1;
	it2Aco=max([ min([indTita2 numTitas]) 1]);
	for indd=[1:length(rangoD)]
		d+=Dinc;
		tita1=tita2;  indTita1=indTita2;  it1Aco=it2Aco;
		tita2=calculaTita(alfaAct,lAct,d);
		indTita2=round((tita2-TitaMin)/TitaInc)+1;
		it2Aco=max([ min([indTita2 numTitas]) 1]);
		usadoIf=0;
		if( (indTita1<=numTitas &&  indTita1>=1) || (indTita2<=numTitas &&  indTita2>=1) )
			usadoIf=1;
			inc=1; if(indTita1>indTita2) inc=-1; endif
			for indTita=[it1Aco:inc:it2Aco]
				#ApuntaEnMatVotaQuien(indd,indTita,indPto,MatVotaQuien,numTitas);
				indMV=(indd-1)*numTitas+indTita;
				posAct=++MatVotaQuien(indMV,1);
				MatVotaQuien(indMV,posAct+1)=indPto;
				usados(indPto)=1; #Se uso
			endfor
		endif
		Traza(indTraza++,:)=[indPto indd d degree(tita1) indTita1 it1Aco degree(tita2) indTita2 it2Aco usadoIf];
	endfor
	if ( usados(indPto)==1 && usadosTeo(indPto)==0 )
		usadosPeroNoTeoricos=[usadosPeroNoTeoricos; indPto];
	endif
	if( usadosTeo(indPto)==1 && usados(indPto)==0 )
		teoricosPeroNoUsados=[teoricosPeroNoUsados; indPto];
	endif
endfor

figure (2);
#mesh(rangoTitas,rangoD,MatVota);

[maxVota,indMaxVota]=max(MatVotaQuien(:,1));
indMaxVota
indDSel=ceil(indMaxVota/numTitas);
indTitaSel=rem(indMaxVota,numTitas);
maxVota;
titaSel=rangoTitas(indTitaSel);
dSel=rangoD(indDSel);
distintos=find(usadosTeo != usados);
numCalculos 
numUsados=length(find(usados))
numUsadosTeo=length(find(usadosTeo))
numDistintos=length(distintos);

disp(["\n 1º (" num2str(dSel) "," num2str(degree(titaSel)) "º) [" int2str(indDSel) \
	"," int2str(indTitaSel) "] con " int2str(maxVota) ])

#Buscamos siguientes maximos
MatVotaQuien2=MatVotaQuien;
indMaxVota2=indMaxVota;
for k=2:5
	MatVotaQuien2(indMaxVota2)=0;
	[maxVota2,indMaxVota2]=max(MatVotaQuien2(:,1));
	indDSel2=ceil(indMaxVota2/numTitas);
	indTitaSel2=rem(indMaxVota2,numTitas)+1;
	titaSel2=rangoTitas(indTitaSel2);
	dSel2=rangoD(indDSel2);
	disp([" " int2str(k) "º (" num2str(dSel2) "," num2str(degree(titaSel2)) "º) [" \
	int2str(indDSel2) "," int2str(indTitaSel2) "] con " int2str(maxVota2) ])
endfor

#Encontramos los que maxTita que contribullen al máximo
contribuyen=MatVotaQuien(indMaxVota,[2:maxVota+1]);
numContribuyen=length(contribuyen)

figure(1);
lr=4;
lUsados=l(find(usados)); alfaUsados=alfa(find(usados));
lContribuyen=l(contribuyen); alfaContribuyen=alfa(contribuyen);
plot(l.*cos(alfa+pi/2),l.*sin(alfa+pi/2)
#  ,lUsados.*cos(alfaUsados+pi/2),lUsados.*sin(alfaUsados+pi/2),"x"
  ,lContribuyen.*cos(alfaContribuyen+pi/2),lContribuyen.*sin(alfaContribuyen+pi/2),"o"
 ,lr*[-cos(titaSel), cos(titaSel)],lr*[-sin(titaSel) ,sin(titaSel)]+dSel
  ,lr*[-cos(TitaMax), 0, cos(TitaMin)],lr*[-sin(TitaMax), 0 ,sin(TitaMin)]+Dmin
  ,lr*[-cos(TitaMin), 0, cos(TitaMax)],lr*[-sin(TitaMin), 0 ,sin(TitaMax)]+Dmax
) ;
grid on


