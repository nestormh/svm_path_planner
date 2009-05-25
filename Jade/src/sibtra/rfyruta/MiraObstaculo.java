/**
 * 
 */
package sibtra.rfyruta;

import sibtra.lms.BarridoAngular;
import sibtra.util.UtilCalculos;

/**
 * Clase para combinar el seguimiento de la ruta con el RF.
 * Proviene de transcribir el código de octave en RangeFinder/branches/RFyGPS/
 * @author alberto
 */
public class MiraObstaculo {

	public static final double PI_2=Math.PI/2;
	
	private boolean debug = true;

	/** Ancho para el camino	 */
	double anchoCamino=2.0;
	
	/** Trayectoria */
	double[][] Tr;

	/** Si la trayectoria es cerrada */
	boolean esCerrada;

	/** Borde Derecho */
	double[][] Bd;
	
	/** Borde Izquierdo */
	double[][] Bi;

	/** Si se encontró indice de inicio por la derecha */
	boolean encontradoInicioD;

	/** Si se encontró indice de inicio por la Izquierda */
	boolean encontradoInicioI;
	
	/** indice punto inicial de búsqueda por Derecha */
	int iptoDini;
	/** indice punto inicial de búsqueda por Izquierda */
	int iptoIini;
	
	/** indice de punto final de búsqueda por la Derecha */
	int iptoD;
	/** indice de punto final de búsqueda por la Izquierda */
	int iptoI;

	/** indice en el barrido donde termino la búsqueda por la Izquierda */
	int iAI;
	/** indice en el barrido donde termino la búsqueda por la Derecha */
	int iAD;

	/** Búsqueda terminó con colisión por la Izquierda */
	boolean ColIzda;
	/** Búsqueda terminó con colisión por la Derecha */
	boolean ColDecha;

	/** posicion de la última invocación a masCercano */
	double[] posActual;

	/** angulo de la última invocación a masCercano */
	double Yaw;

	/** indice en barrido donde se dió la mínima distancia */
	int indMin;

	/** barrido en la última invocación a masCercano */
	BarridoAngular barr;

	/** Distancia lineal a la que se encuentra el obstáculo teóricamente más cercano */
	double dist;
	
	/** Distancia en camino donde está el obstáculo más cercano (sobre camino).
	 * Será MAX_VALUE si no se ha encontrado
	 */
	double distCamino;

	/** indice del segmento de la trayectoria donde está el coche */
	int indiceCoche=-1;
	
	int iLibre;

	/** indice del segmento de la trayectoria anterior al obstáculo*/
	int indSegObs;

	/** indice del barrido donde está el obstáculo más cercano en el camino*/
	int indBarrSegObs;

	/** Si se encontró el segmento donde está el obstáculo */
	boolean encontradoSegObs;

	/** Si se ha invocado a mas cercano con datos válidos */
	boolean hayDatos=false;
	
	/** Se activa cuando esta fuera del camino */
	boolean estaFuera=true;

	/**
	 * Constructor necesita conocer la ruta que se va a seguir.
	 * A partir de ella generará los bordes de la carretera
	 * @param trayectoria en coordenadas locales.
	 * @param si la trayectoria es cerrada o no.
	 * @param debug si queremos activar mensajes de depuracion
	 */
	public MiraObstaculo(double[][] trayectoria, boolean esCerrada, boolean debug) {
		this.esCerrada=esCerrada;
		this.debug=debug;
		if(trayectoria==null || trayectoria.length<2 || trayectoria[1].length<2)
			throw (new IllegalArgumentException("Trayectoria no tienen las dimensiones mínimas"));
		Tr=trayectoria;
		construyeCamino(Tr, anchoCamino);
		indiceCoche=0;
				
	}

	/** Constructor sin debug */
	public MiraObstaculo(double[][] trayectoria, boolean esCerrada) {
		this(trayectoria, esCerrada, false);
	}
	
	/** Constructor sin debug y trayectoria supuesta abierta*/
	public MiraObstaculo(double[][] trayectoria) {
		this(trayectoria, false, false);
	}
	
	
	/** @return the debug */
	public boolean isDebug() {
		return debug;
	}

	/** @param debug the debug to set */
	public void setDebug(boolean debug) {
		this.debug = debug;
	}

	/**
	 * Transcripción de código octave RangeFinder/branches/RFyGPS/ConstruyeCamino.m
	 * @param tr trayectoria
	 * @param ancho ancho del camino
	 */
	private void construyeCamino(double[][] tr, double ancho) {
		Bi=new double[tr.length][2];
		Bd=new double[tr.length][2];

		//Si es cerrada todos los puntos se calculan en el bucle
		//Si no es cerrada los primeros y últimos puntos se calculan fuera del bucle.
		//Las variables se inicializan como si NO fuera cerrada.
		int indIni=1;
		int indFin=tr.length-1;
		
		//Primeros puntos de los bordes a partir del los 2 primeros puntos
		double[] p1=tr[0];
		double[] p2=tr[1];
		double[] v2= {p2[0]-p1[0],p2[1]-p1[1]};
		if( !esCerrada) {	
			//los 2 primeros puntos
			double modV2=Math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]);
			//unitario girado 90º
			double[] v2u90={-v2[1]/modV2, v2[0]/modV2};

			Bi[0][0]=p1[0]+v2u90[0]*ancho;  Bi[0][1]=p1[1]+v2u90[1]*ancho;
			Bd[0][0]=p1[0]-v2u90[0]*ancho;	Bd[0][1]=p1[1]-v2u90[1]*ancho;
		} else {
			//Se cambian las variables para el caso de cerrada
			indIni=0;
			indFin=tr.length;
			p1=tr[tr.length-1];
			p2=tr[0];
			v2[0]=p2[0]-p1[0];	v2[1]=p2[1]-p1[1];
		}
		for(int i=indIni; i<indFin; i++) {
			p1=p2;
			p2=tr[(i+1)%tr.length]; //ciclamos por si es cerrada.
			double[] v1={v2[0],v2[1]};
			v2[0]=p2[0]-p1[0];	v2[1]=p2[1]-p1[1];
			
			double Ti1=Math.atan2(v1[1], v1[0]);
			//Tita medios
			double Ti_2=UtilCalculos.anguloVectores(v1,v2)/2.0;
			double d=ancho/Math.cos(Ti_2);
			double angRot=Ti1-PI_2+Ti_2;
			//vector perpendicular medio a v1 y v2
			double[] vpc={d*Math.cos(angRot),	d*Math.sin(angRot)};
			
			
			//Añadimos un punto a cada uno de los bordes
			Bd[i][0]=p1[0]+vpc[0];	Bd[i][1]=p1[1]+vpc[1];
			Bi[i][0]=p1[0]-vpc[0];	Bi[i][1]=p1[1]-vpc[1];
			
		}
		if (!esCerrada){
			//los 2 últimos puntos
			double modV2=Math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]);
			//unitario girado 90º
			double[] v2u90={-v2[1]/modV2, v2[0]/modV2};

			Bi[tr.length-1][0]=p2[0]+v2u90[0]*ancho;	Bi[tr.length-1][1]=p2[1]+v2u90[1]*ancho;
			Bd[tr.length-1][0]=p2[0]-v2u90[0]*ancho;	Bd[tr.length-1][1]=p2[1]-v2u90[1]*ancho;
		}		
	}
	
	/** 
	 * Se debe invocar cuando la posición del coche a cambiado mucho desde la última invocación
	 * a {@link #masCercano(double[], double, BarridoAngular)}.
	 * Se pone {@link #indiceCoche} a valor -1 para que la búsqueda del punto más cercano
	 * se haga en toda la ruta.
	 */
	public void nuevaPosicion() {
		indiceCoche=-1;
	}
	
	/**
	 * En un momento dado nos dice a que distancia se encuentra el obstaculo más cercano
	 * @param posicionLocal Posición en coordenadas locales donde nos encontramos
	 * @param yawA rumbo actual del vehiculo hacia el norte (EN RADIANES)
	 * @param barrAct último barrido angular
	 * @return Distancia libre en el camino. 
	 */
	public double masCercano(double[] posicionLocal, double yawA, BarridoAngular barrAct) {
		barr=barrAct;
		posActual=posicionLocal;
		Yaw=yawA; //si está ya en radianes, si no tendríamos que hacerer el cambio.
		dist=Double.NaN; //por si fallan las primeras comprobaciones

		//Inicializamos todas las variables del algoritmo
		ColIzda=false; //colisión por la izda.
		ColDecha=false; //colisión por la derecha
		iAI=barrAct.numDatos()-1; iAD=0;
		hayDatos=true;
		double AngD=barrAct.getAngulo(iAD);
		double AngI=barrAct.getAngulo(iAI);
		double resAng=Math.toRadians(barr.incAngular*0.25);
		double AngIniB=barrAct.getAngulo(0);  //angulo inicial del barrido
		double AngFinB=barrAct.getAngulo(barrAct.numDatos()-1); //angulo final del barrido
		
		//Si en la iteración anterior estabamos fuera buscamos desde el principio
		//TODO no funciona bie, corregir
//		if(estaFuera) indiceCoche=-1;
		//punto de la trayectoria más cercano a la posición local
		indiceCoche=UtilCalculos.indiceMasCercanoOptimizado(Tr, esCerrada, posicionLocal, indiceCoche);
		
		double distATr=distanciaPuntos(posicionLocal, Tr[indiceCoche]);
		//El coche solo puede estar en el segmento del más cercano o en el anterior
		if ((indiceCoche==(Tr.length-1) && !esCerrada) 
				|| !dentroSegmento(posicionLocal, indiceCoche))
			//no puede estar en indiceCoche
			//Tenemos que ver si está en el anterior
			if(indiceCoche==0 && !esCerrada) {
				//No hay opción al anterior
				log("Cochoe está antes del primer punto");
				return dist; //NaN
			} else {
				indiceCoche=(indiceCoche+Tr.length-1)%Tr.length; //anterior generalizado
				if (!dentroSegmento(posicionLocal, indiceCoche)) {
					if(distATr<=anchoCamino)
						System.err.println("No está tampoco en segmento anterior a pesar de distacia < camino");
					//esta fuera del camino
					estaFuera=true;
					return dist; //NaN					
				}
			}
		estaFuera=false;
		indiceCoche++;
		//inidiceCoche tiene en ídice del siguiente al segmento, por lo que el coche estará un poquito más atras		
		
		double[] v={Math.cos(Yaw), Math.sin(Yaw)};  //vector con la orientación del coche
		
		//vamos a ver si estamos yendo hacia atrás
		double[] vCamino={Tr[indiceCoche][0]-Tr[indiceCoche-1][0],
				Tr[indiceCoche][1]-Tr[indiceCoche-1][1] }; //vector en la dirección del camino
		double difDir=UtilCalculos.anguloVectores(vCamino, v);
		if(Math.abs(difDir)>(PI_2)) {
			System.out.println("Estamos yendo hacia atrás");
			return dist; //salimos 
		}

		//Tenemos que buscar que punto del borde derecho e izquierdo comienza el barrido
		//por un lado y el otro.
		{	//Comenzamos por la derecha
			iptoDini=indiceCoche; //comenzamos a probar con en pto delante de donde está el coche
			double[] vD={Bd[iptoDini][0]-posicionLocal[0],	Bd[iptoDini][1]-posicionLocal[1]};
			double angD=UtilCalculos.anguloVectores(v, vD);
			boolean enRango=(Math.abs(angD)<=PI_2 && (angD+PI_2)>=AngIniB);
			if(Math.abs(angD)>PI_2 || (angD+PI_2)<AngIniB) {
				//punto por detrás del barrido, tenemos que avanzar para encontrarlo
				boolean alcanzable,seFueCamino;
				double[] vI=new double[2];
				double angI;
				do {
					iptoDini=(iptoDini+1)%Bd.length; //incrementamos ciclando por si es cerrada
					vD[0]=Bd[iptoDini][0]-posicionLocal[0];	vD[1]=Bd[iptoDini][1]-posicionLocal[1];
					angD=UtilCalculos.anguloVectores(v, vD);
					enRango=(Math.abs(angD)<=PI_2 && (angD+PI_2)>=AngIniB);
					//vemos si el punto del borde izquierdo se fue del rango del RF
					vI[0]=Bi[iptoDini][0]-posicionLocal[0]; vI[1]=Bi[iptoDini][1]-posicionLocal[1];
					angI=UtilCalculos.anguloVectores(v, vI);
					seFueCamino=Math.abs(angI)>PI_2 ||  (angI+PI_2)<AngIniB;
				} while(
						(alcanzable=(largoVector(vD)<=barrAct.getDistanciaMaxima())) //se alcance
						&& !enRango //no hayamos entrado en rango
						&& !seFueCamino //no se haya ido todo el camino del rango del RF
						&& (esCerrada || iptoDini<(Bd.length-1)) //no se haya acabado la trayectoria
						&& iptoDini!=indiceCoche  //no hayamos dado toda la vuelta
				);
				//encontrado si, es alcanzable y estamos en el rango
				encontradoInicioD=alcanzable && enRango;
				if(seFueCamino) log("Se fue todo el camino buscando por la Derecha");
			} else {
				//No tiene sentido retrasarse
				encontradoInicioD=true; //Siempre tendremos inicio (alcance barrido > ancho del camino)
			}
		}
		{	//Ahora la izquierda
			double[] v2=new double[2];
			iptoIini=indiceCoche; //comenzamos a probar con en pto delante de donde está el coche
			v2[0]=Bi[iptoIini][0]-posicionLocal[0];	v2[1]=Bi[iptoIini][1]-posicionLocal[1];
			double angI=UtilCalculos.anguloVectores(v, v2);
			boolean enRango=(Math.abs(angI)<=PI_2 && (angI+PI_2)<=AngFinB);
			if(Math.abs(angI)>PI_2 || (angI+PI_2)>AngFinB) {
				//punto por detrás del barrido, tenemos que avanzar para encontrarlo
				boolean alcanzable,seFueCamino;
				double[] vD=new double[2];
				double angD;
				do {
					iptoIini=(iptoIini+1)%Bi.length; //incrementamos ciclando por si es cerrada
					v2[0]=Bi[iptoIini][0]-posicionLocal[0];	v2[1]=Bi[iptoIini][1]-posicionLocal[1];
					angI=UtilCalculos.anguloVectores(v, v2);
					enRango=(Math.abs(angI)<=PI_2 && (angI+PI_2)<=AngFinB);
					//vemos si el punto del borde derecho se fue del rango del RF
					vD[0]=Bd[iptoIini][0]-posicionLocal[0]; vD[1]=Bd[iptoIini][1]-posicionLocal[1];
					angD=UtilCalculos.anguloVectores(v, vD);
					seFueCamino=Math.abs(angD)>PI_2 ||  (angD+PI_2)>AngFinB;
				} while(
						(alcanzable=(largoVector(v2)<=barrAct.getDistanciaMaxima()))
						&& !enRango
						&& !seFueCamino //no se haya ido todo el camino del rango del RF						
						&& (esCerrada || iptoIini<(Bi.length-1))
						&& iptoIini!=indiceCoche  //no hayamos dado toda la vuelve
				);
				encontradoInicioI=alcanzable && enRango;
				if(seFueCamino) log("Se fue todo el camino buscando por la Izquierda");
			} else {
				//No tiene sentido retrasarse
				encontradoInicioI=true; //Siempre tendremos inicio (alcance barrido > ancho del camino)
			}
		}	
		
		if(!encontradoInicioD && !encontradoInicioI) {
			System.err.println("No se ha encontrado ningún inicio !!!");
			return dist; //NaN
		}
		
		//procedemos a recorrer los puntos para ver si hay algo dentro
		boolean avanD=encontradoInicioD; // Si se avanza la derecha
		boolean avanI=encontradoInicioI; //Si se avanza le izquierda
		iptoI=(avanI?iptoIini:0); //0 si no se ha iniciado
		iptoD=(avanD?iptoDini:0); //0 si no se ha iniciado
		iptoI--;  iptoD--; //se incrementan al entrar en el bucle
		while (avanD) {
			if(!esCerrada && iptoD==(Bd.length-1)) {
				log("No queda borde derecho, no podemos seguir avanzando");
				avanD=false;
			} else {
				//avanzamos por la derecha
				iptoD=(iptoD+1)%Bd.length; //incrementamos ciclando por si es cerrada
				double AngDant=AngD;
				double[] v2D={Bd[iptoD][0]-posicionLocal[0], Bd[iptoD][1]-posicionLocal[1]};
				double DistD=largoVector(v2D);
				AngD=UtilCalculos.anguloVectores(v, v2D)+PI_2; //ya que 0º está 90º a la derecha
				if (AngD<AngDant) {
					log("El camino gira Decha Dejamos derecha");
					avanD=false;
				} else {
					//Vemos distancia barrido en angulo encontrado
					iAD=(int)Math.floor((AngD-AngIniB)/resAng);
					if(iAD>=barr.numDatos()) {
						//el ángulo de este punto del borde es mayor que el barrido
						log("No queda barrido para pto Decha");
						iAD=barr.numDatos()-1; 
						avanD=false;						
					} else if(barr.getDistancia(iAD)<barr.getDistanciaMaxima() 
							&& DistD>barr.getDistancia(iAD)) {
						//el angulo del barrido anterior colisiona
						ColDecha=true;
						avanD=false;						
					} else if (iAD<(barr.numDatos()-1) 
							&& barr.getDistancia(iAD+1)<barr.getDistanciaMaxima() 
							&& DistD>barr.getDistancia(iAD+1)) {
						//el angulo del barrido siguiente colisiona
						iAD++;  
						ColDecha=true;
						avanD=false;
					}
				}
			}
		}
		while (avanI) {
			if(!esCerrada && iptoI==(Bi.length-1)) {
				log("No queda borde izdo, no podemos seguir avanzando"	);
				avanI=false;
			} else {
				//avanzamos por la izda.
				iptoI=(iptoI+1)%Bi.length; //incrementamos ciclando por si es cerrada
				double AngIant=AngI;
				double[] v2I={Bi[iptoI][0]-posicionLocal[0], Bi[iptoI][1]-posicionLocal[1]};
				double DistI=largoVector(v2I);
				AngI=UtilCalculos.anguloVectores(v, v2I)+PI_2; //ya que 0º está 90º a la derecha
				if (AngI>AngIant) {
					log("El camino gira Izquierda. Dejamos izquierda");
					avanI=false;
				} else {
					//Vemos distancia barrido en angulo encontrado
					iAI=(int)Math.ceil((AngI-AngIniB)/resAng);
					if(iAI<0) {
						//el ángulo de este punto del borde es menor que el barrido
						log("No queda barrido para pto Izda");
						iAI=0;
						avanI=false;
					} else if(barr.getDistancia(iAI)<barr.getDistanciaMaxima() 
							&& DistI>barr.getDistancia(iAI)) {
						//el angulo del barrido anterior colisiona
						ColIzda=true;
						avanI=false;						
					} else if ((iAI>0) 
							&& barr.getDistancia(iAI-1)<barr.getDistanciaMaxima() 
							&& DistI>barr.getDistancia(iAI-1)) {
						//el angulo del barrido siguiente colisiona
						iAI--;
						ColIzda=true;
						avanI=false;
					}
				}
			}
		}
			
		indSegObs=Integer.MAX_VALUE;
		indBarrSegObs=Integer.MAX_VALUE;
		if(encontradoInicioD && encontradoInicioI  //se iniciado ambos lados, si no no tienen sentido usar iAD ó iAI 
				&& (iAD<iAI || (ColDecha && ColIzda))) {
			log("Los rayos no se han cruzado o hay 2 colisiones");
			if(!ColDecha && !ColIzda) {
				System.err.println("No se han cruzado y no hay colisión ??");
			}
			//buscamos posible minimo en medio
			indMin=iAD;
			for(int i=iAD+1; i<=iAI; i++)
				if(barr.getDistancia(i)<barr.getDistancia(indMin))
					indMin=i;
			dist=barr.getDistancia(indMin);
			//Tenemos que buscar pto dentro del camino más cercano
			if(!buscaSegmentoObstaculo(posicionLocal,iAD, iAI,indMin))
				//usamos hasta donde hemos podido explorar
				indSegObs=(iptoD<iptoI)?iptoI:iptoD;
		} else {
			log("los rayos se han cruzado y no hay colisión en los 2 o alguno no se a iniciado");
			if(ColIzda) {
				//usamos el punto de la trayectoria hasta donde podemos llegar
				dist=-distanciaPuntos(Tr[iLibre=iptoI],posActual);
				//Tenemos que buscar pto dentro del camino más cercano
				if(!buscaSegmentoObstaculo(posicionLocal, 0, iAI))
					//usamos hasta donde hemos podido explorar
					indSegObs=iptoI;
			} else  if(ColDecha) {
				dist=-distanciaPuntos(Tr[iLibre=iptoD],posActual);
				//Tenemos que buscar pto dentro del camino más cercano
				if(!buscaSegmentoObstaculo(posicionLocal, iAD, barr.numDatos()-1))
					//usamos hasta donde hemos podido explorar
					indSegObs=iptoD;
			} else {
				//no ha colisionado ningún lado, cogemos el índice mayor
				// tambien para el camino
				//Si alguno no iniciado su índice será negativo
				iLibre=indSegObs=(iptoD<iptoI)?iptoI:iptoD;
				indBarrSegObs=Integer.MAX_VALUE;
				dist=-distanciaPuntos(Tr[iLibre],posActual);
				encontradoSegObs=false;
			}
		}
			
		log("Indice del segmento coche "+(indiceCoche-1)
				+" anterior al obstáculo ("+encontradoSegObs+")"+indSegObs);
		distCamino=largoTramo(indiceCoche,indSegObs);
		return distCamino;
	}

	
	private int segmentoObstaculoParaIndice (double[] posicionLocal, int indiceBarrido,int incSegObs) {
		double angI=barr.getAngulo(indiceBarrido);
		double distI=barr.getDistancia(indiceBarrido);
		double[] ptoI={posicionLocal[0]+distI*Math.cos(Yaw+angI-PI_2)
				,posicionLocal[1]+distI*Math.sin(Yaw+angI-PI_2)};
		//Buscamos segmento en que está
		//TODO usar algo más eficiente que la fuerza bruta
		int incSA=0;
		boolean enseg=false;
		while (incSA<incSegObs  //no buscamos más allá de segmento ya encontrado
				&& !(enseg=dentroSegmento(ptoI, (indiceCoche+incSA)%Tr.length)) 
				) {
			incSA++;  //no está, vamos avanzando
		} 
		if(enseg) {
			incSegObs=incSA; //se encontró uno más cercano
			indBarrSegObs=indiceBarrido;
		}
		return incSegObs;
	}
	
	/** Para la posición local actual, busca un rango de barridos del RF, para determinar el segmento de la trayectoria 
	 * más cercano en que se da una colisión.
	 * @param posicionLocal
	 * @param indComBarrido 
	 * @param indFinBarrido
	 * @param indiceMasCercano pista de en que índice del barrido puede estar el más cercano, para reducir la búsqueda
	 * @return si alguno de los puntos dio colisión dentro del camino.
	 */
	private boolean buscaSegmentoObstaculo(double[] posicionLocal, int indComBarrido,int indFinBarrido, int indiceMasCercano) {
		int incSegObs; //incremento sobre la posición del coche donde se encuentra el ostaculo 
		incSegObs=esCerrada?Tr.length-1:Tr.length-indiceCoche-1; //para limitar la búsquda
		indBarrSegObs=Integer.MAX_VALUE;
		encontradoSegObs=false;
		int nuevoIncSegObst;
		//si es válido indiceMasCercano probamos primero con ese índice (tendremos el mínimo más rápido)
		if(indiceMasCercano>=indComBarrido && indiceMasCercano<=indFinBarrido) {
			nuevoIncSegObst=segmentoObstaculoParaIndice(posicionLocal, indiceMasCercano, incSegObs);
			encontradoSegObs=encontradoSegObs||(nuevoIncSegObst!=incSegObs); //desde que se encuentre algún segmento pasará a true
			incSegObs=nuevoIncSegObst;
		}
		for(int i=indComBarrido; i<=indFinBarrido; i++) {
			nuevoIncSegObst=segmentoObstaculoParaIndice(posicionLocal, i, incSegObs);
			encontradoSegObs=encontradoSegObs||(nuevoIncSegObst!=incSegObs); //desde que se encuentre algún segmento pasará a true
			incSegObs=nuevoIncSegObst;
		}
		indSegObs=((indiceCoche)+incSegObs)%Tr.length;
		if(!encontradoSegObs)  {
			log("No se encontró segmento ");
			indBarrSegObs=Integer.MAX_VALUE;
		}
		return encontradoSegObs;
	}
	
	/** Ídem {@link #buscaSegmentoObstaculo(double[], int, int, int)} pero sin pista sobre el más cercano */
	private boolean buscaSegmentoObstaculo(double[] posicionLocal, int indComBarrido,int indFinBarrido) {
		return buscaSegmentoObstaculo(posicionLocal, indComBarrido, indFinBarrido, -1);
	}
	
	
	private void log(String string) {
		if(debug)
			System.out.println(string);		
	}

	/** @return largo euclídeo del vector */
	public static double largoVector(double[] d) {
		return Math.sqrt(d[0]*d[0]+d[1]*d[1]);
	}

	/** @return distancia ecuclídea entre p1 y p2	 */
	public static double distanciaPuntos(double[] p1, double[] p2) {
		double[] d={p1[0]-p2[0], p1[1]-p2[1]};
		return largoVector(d);
	}

	
	/**
	 * Dice si pto pasado esta en cuadrilátero de la trayectoria.
	 * Lo hacemos dividiendo cuadrilátero en dos triangulos y determiando si punto pertenece a alguno de los triangulos.
	 * Sacado de  http://gmc.yoyogames.com/index.php?showtopic=409110
	 * La explicación de la pertenencia a triangulo, sacada de http://2000clicks.com/MathHelp/GeometryPointAndTriangle2.htm
	 * @param pto por el que se pregunta
	 * @param i cuadrilátero i-ésimo del camino
	 * @return si está dentro
	 */
	public boolean dentroSegmento(double[] pto,int i) {
		if(i<0)
			throw new IllegalArgumentException("Pasado indice negativo");
		if(i>=Tr.length)
			throw new IllegalArgumentException("Indice supera largo trayectoria");
		if(!esCerrada && i==(Tr.length-1))
			throw new IllegalArgumentException("Es abierta y se a pasado úlitmo indice válido");
		int psig=i+1;
		if(esCerrada && i==(Tr.length-1)) 
			psig=0;
		
		//usamos directamente el área con sentido en todo el cuadrilátero.
		/*Cuadrilatero 
		 *  D=Bi[i+1] * --------------------------------* C=Bd[i+1] 
		 *            |                                 |
		 *            |                                 |
		 *            |                                 |
		 *  A=Bi[i]   * --------------------------------* B=Bd[i] 
		 *  
		 */
//		log("esq=["
//				+Bi[i][0]+","+Bi[i][1]+";"
//				+Bi[psig][0]+","+Bi[psig][1]+";"
//				+Bd[psig][0]+","+Bd[psig][1]+";"
//				+Bd[i][0]+","+Bd[i][1]+";"
//				+"], pto=["+pto[0]+","+pto[1]+"]"
//				);
		

		//Las ordenamos para intentar que se dispare primero la más probable y evitar más cálculos
		//Suponemos que lo más probable es que esté delante (falla fCD), detrás (fall fAB) y luego los lados
		//(no supone una gran mejora).
		return 
		((pto[1]-Bd[psig][1])*(Bi[psig][0]-Bd[psig][0])-(pto[0]-Bd[psig][0])*(Bi[psig][1]-Bd[psig][1]))>0 //fCD si está sobre el segmento BD, no lo consideramos (será considerado por el siguiente)
		&& 
		((pto[1]-Bi[i][1])*(Bd[i][0]-Bi[i][0])-(pto[0]-Bi[i][0])*(Bd[i][1]-Bi[i][1]))>=-1e-10 //fAB 
		&& 
		((pto[1]-Bd[i][1])*(Bd[psig][0]-Bd[i][0])-(pto[0]-Bd[i][0])*(Bd[psig][1]-Bd[i][1]))>=-1e-10 //fBC 
		&& 
		((pto[1]-Bi[psig][1])*(Bi[i][0]-Bi[psig][0])-(pto[0]-Bi[psig][0])*(Bi[i][1]-Bi[psig][1]))>=-1e-10 //fDA
		;
	}
	
//	/**
//	 * Dice si pto pasado esta en cuadrilátero de la trayectoria
//	 * @param pto por el que se pregunt
//	 * @param i cuadrilátero i-ésimo del camino
//	 * @return si está dentro
//	 */
//	public boolean dentroSegmentoAproximado(double[] pto,int i){
//		if(i<0)
//			throw new IllegalArgumentException("Pasado indice negativo");
//		if(i>=Tr.length)
//			throw new IllegalArgumentException("Indice supera largo trayectoria");
//		if(!esCerrada && i==(Tr.length-1))
//			throw new IllegalArgumentException("Es abierta y se a pasado úlitmo indice válido");
//		int psig=i+1;
//		if(esCerrada && i==(Tr.length-1)) psig=0;
//		int pant=i-1;
//		if(esCerrada && i==0) pant=Tr.length-1;
//		
//		double distI=distanciaPuntos(pto, Tr[i]);
//		double distI1=distanciaPuntos(pto, Tr[psig]);
//		
//		if(distI<=anchoCamino && distI1<=anchoCamino)
//			return true;
//		
//		return false;	
//	}	
	
	
	public double largoTramo(int iini, int ifin) {
		if(iini<0 || iini>=Tr.length)
			throw new IllegalArgumentException("Inidice inicial fuera de rango válido "+iini);
		if(ifin<0 || ifin>=Tr.length)
			throw new IllegalArgumentException("Inidice final fuera de rango válido "+ifin);
		if(!esCerrada && iini>ifin)
			throw new IllegalArgumentException("No es cerrada y indice inicial ("+iini+") > inidice final ("+ifin+")");
		
		if(!esCerrada && iini>ifin || ifin>=Tr.length)
			return Double.POSITIVE_INFINITY;
		double largo=0;
		for(int i=iini; i!=ifin; i=(i+1)%Tr.length) {
			double[] v={Tr[(i+1)%Tr.length][0]-Tr[i][0], Tr[(i+1)%Tr.length][1]-Tr[i][1]}; 
			largo+=largoVector(v);
		}
		return largo;
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		double[][] Tr={{ 18.656355 , 1.793361}
		,{ 19.540099 , 1.855596}
		,{ 20.247094 , 1.905385}
		,{ 21.163414 , 2.129761}
		,{ 21.870409 , 2.179549}
		,{ 22.754152 , 2.241785}
		,{ 23.461147 , 2.291573}
		,{ 24.344890 , 2.353809}
		,{ 25.084462 , 2.565738}
		,{ 25.968205 , 2.627974}
		,{ 26.675200 , 2.677762}
		,{ 27.558943 , 2.739998}
		,{ 28.265937 , 2.789787}
		,{ 28.972932 , 2.839576}
		,{ 29.889252 , 3.063952}
		,{ 30.628824 , 3.275880}
		,{ 31.368396 , 3.487809}
		,{ 32.140545 , 3.861878}
		,{ 32.735946 , 4.223499}
		,{ 33.508095 , 4.597568}
		,{ 33.959325 , 5.108882}
		,{ 34.554725 , 5.470504}
		,{ 35.071110 , 6.306098}
		,{ 35.587495 , 7.141692}
		,{ 35.894553 , 7.802700}
		,{ 36.234190 , 8.625847}
		,{ 36.364500 , 9.274407}
		,{ 36.318061 , 9.910520}
		,{ 36.271622 , 10.546633}
		,{ 36.081012 , 11.332439}
		,{ 35.857824 , 11.956105}
		,{ 35.602059 , 12.417631}
		,{ 35.136968 , 12.704570}
		,{ 34.527705 , 13.141201}
		,{ 34.062614 , 13.428140}
		,{ 33.597522 , 13.715078}
		,{ 32.746356 , 13.814982}
		,{ 32.071938 , 13.927334}
		,{ 31.397520 , 14.039685}
		,{ 30.690525 , 13.989896}
		,{ 29.983529 , 13.940107}
		,{ 29.276534 , 13.890319}
		,{ 28.569539 , 13.840530}
		,{ 27.862543 , 13.790741}
		,{ 27.155548 , 13.740952}
		,{ 26.239226 , 13.516576}
		,{ 25.532231 , 13.466787}
		,{ 24.825235 , 13.416999}
		,{ 24.118240 , 13.367210}
		,{ 23.201918 , 13.142834}
		,{ 22.494923 , 13.093046}
		};
		
		MiraObstaculo MI=new MiraObstaculo(Tr);
		
		System.out.println("Bdj=[");
		for(int i=0; i<MI.Bd.length; i++)
			System.out.println(MI.Bd[i][0]+","+MI.Bd[i][1]);
		System.out.println("];");
		
		System.out.println("Bij=[");
		for(int i=0; i<MI.Bi.length; i++)
			System.out.println(MI.Bi[i][0]+","+MI.Bi[i][1]);
		System.out.println("];");
	}

	/**
	 * @return el bd
	 */
	public double[][] getBd() {
		return Bd;
	}

	/**
	 * @return el bi
	 */
	public double[][] getBi() {
		return Bi;
	}

	/**
	 * @return el tr
	 */
	public double[][] getTr() {
		return Tr;
	}

	
	public String toString() {
		String ret="Lineal="+dist+" camino="+distCamino;
		ret+=" \n indiceCoche ="+indiceCoche
			+" encontradoInicioD:"+encontradoInicioD +"  iptoDini ="+iptoDini
			+" encontradoInicioI:"+encontradoInicioI +"  iptoIini ="+ iptoIini
			+"\n iAD="+iAD
			+" ColDecha =" + ColDecha
			+" iptoD ="+iptoD
				+"\n iAI="+iAI
				+" ColIzda ="+ ColIzda
				+"  iptoI ="+iptoI
				+" \n imin ="+indMin
				+" indSegObs ="+indSegObs
				+" indBarrSegObs ("+encontradoSegObs+") ="+indBarrSegObs
		;
		return ret;
	}
}
