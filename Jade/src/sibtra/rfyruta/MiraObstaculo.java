/**
 * 
 */
package sibtra.rfyruta;

import sibtra.lms.BarridoAngular;

/**
 * Clase para combinar el seguimiento de la ruta con el RF.
 * Proviene de transcribir el código de octave en RangeFinder/branches/RFyGPS/
 * @author alberto
 */
public class MiraObstaculo {
	
	private boolean debug = true;

	/** Ancho para el camino	 */
	double anchoCamino=2.0;
	
	/** Trayectoria */
	double[][] Tr;
	
	/** Borde Derecho */
	double[][] Bd;
	
	/** Borde Izquierdo */
	double[][] Bi;

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

	/** Índice del punto de la trayectoria que está cerca por detrás */
	int indiceDentro;

	/** indice del segmento de la trayectoria donde está el coche */
	int indiceCoche;
	
	int iLibre;

	/** indice del segmento de la trayectoria anterior al obstáculo*/
	int indSegObs;

	/** indice del barrido donde está el obstáculo más cercano en el camino*/
	int indBarrSegObs;
	
	/**
	 * Constructor necesita conocer la ruta que se va a seguir.
	 * A partir de ella generará los bordes de la carretera
	 * @param trayectoria en coordenadas locales.
	 * @param debug si queremos activar mensajes de depuracion
	 */
	public MiraObstaculo(double[][] trayectoria, boolean debug) {
		this.debug=debug;
		if(trayectoria==null || trayectoria.length<2 || trayectoria[1].length<2)
			throw (new IllegalArgumentException("Trayectoria no tienen las dimensiones mínimas"));
		Tr=trayectoria;
		construyeCamino(Tr, anchoCamino);
		indiceDentro=0;
				
	}

	/** Constructor sin debug */
	public MiraObstaculo(double[][] trayectoria) {
		this(trayectoria,false);
	}
	/**
	 * Transcripción de código octave RangeFinder/branches/RFyGPS/ConstruyeCamino.m
	 * @param tr trayectoria
	 * @param ancho ancho del camino
	 */
	private void construyeCamino(double[][] tr, double ancho) {
		Bi=new double[tr.length][2];
		Bd=new double[tr.length][2];

		//Primeros puntos de los bordes a partir del los 2 primeros puntos
		double[] p1=tr[0];
		double[] p2=tr[1];
		double[] v2= {p2[0]-p1[0],p2[1]-p1[1]};
		double modV2=Math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]);
		//unitario girado 90º
		double[] v2u90={-v2[1]/modV2, v2[0]/modV2};
		
		Bi[0][0]=p1[0]+v2u90[0]*ancho;  Bi[0][1]=p1[1]+v2u90[1]*ancho;
		Bd[0][0]=p1[0]-v2u90[0]*ancho;	Bd[0][1]=p1[1]-v2u90[1]*ancho;
		
		for(int i=1; i<tr.length-1; i++) {
			double[] p0=p1;
			p1=p2;
			p2=tr[i+1];
			double[] v1={v2[0],v2[1]};
			v2[0]=p2[0]-p1[0];	v2[1]=p2[1]-p1[1];
			double[] v1u90=v2u90;
			modV2=Math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]);
			v2u90[0]=-v2[1]/modV2;	v2u90[1]=v2[0]/modV2;
			
			double Ti1=Math.atan2(v1[1], v1[0]);
			//Tita medios
			double Ti_2=anguloVectores(v1,v2)/2.0;
			double d=ancho/Math.cos(Ti_2);
			double angRot=Ti1-Math.PI/2+Ti_2;
			//vector perpendicular medio a v1 y v2
			double[] vpc={d*Math.cos(angRot),	d*Math.sin(angRot)};
			
			Bd[i][0]=p1[0]+vpc[0];	Bd[i][1]=p1[1]+vpc[1];
			Bi[i][0]=p1[0]-vpc[0];	Bi[i][1]=p1[1]-vpc[1];
			
		}
		Bi[tr.length-1][0]=p2[0]+v2u90[0]*ancho;	Bi[tr.length-1][1]=p2[1]+v2u90[1]*ancho;
		Bd[tr.length-1][0]=p2[0]-v2u90[0]*ancho;	Bd[tr.length-1][1]=p2[1]-v2u90[1]*ancho;
		
	}
	
	/**
	 * @param v1 primer verctor de 2 componentes (x,y)
	 * @param v2 segundo vector de 2 componenetes (x,y)
	 * @return angulo formado por los 2 vectores en rango (-PI,PI)
	 */
	public static double anguloVectores(double[] v1, double[] v2) {
		double Ti1=Math.atan2(v1[1],v1[0]);
		double Ti2=Math.atan2(v2[1],v2[0]);
		double Ti=Ti2-Ti1;
		if(Ti<=-Math.PI)
			Ti=Ti+2*Math.PI;
		if(Ti>Math.PI)
			Ti=Ti-2*Math.PI;
		return Ti;
	}

	/**
	 * En un momento dado nos dice a que distancia se encuentra el obstaculo más cercano
	 * @param posicionLocal Posición en coordenadas locales donde nos encontramos
	 * @param rumbo actual del vehiculo hacia el norte (EN RADIANES)
	 * @return Distancia libre en el camino. 
	 */
	public double masCercano(double[] posicionLocal, double yawA, BarridoAngular barrAct) {
		barr=barrAct;
		posActual=posicionLocal;
		Yaw=yawA; //si está ya en radianes, si no tendríamos que hacerer el cambio.
		dist=Double.NaN; //por si fallan las primeras comprobaciones

		//Inicializamos todas las variables del algoritmo
		boolean avanD=true; //avanza la derecha
		boolean avanI=true; //avanza le izquierda
		ColIzda=false; //colisión por la izda.
		ColDecha=false; //colisión por la derecha
		iAI=barrAct.numDatos()-1; iAD=0;
		double AngD=barrAct.getAngulo(iAD);
		double AngI=barrAct.getAngulo(iAI);
		double resAng=Math.toRadians(barr.incAngular*0.25);
		//¿Puedo saber si estoy dentro del camino??
		//  Basta que la distancia a alguno de los puntos del camino sea menor que Ancho.
		//comienzo por el indice de centro anterior
		if(distanciaPuntos(posicionLocal,Tr[indiceDentro])<anchoCamino)  {
			//el anterior ya esta cerca, busco hacia popa (atrás) hasta salir
			while(indiceDentro>=0 
					&& distanciaPuntos(posicionLocal,Tr[indiceDentro])<anchoCamino)
				indiceDentro--;
			indiceDentro++; //nos quedamos con el siguiente (que si está cerca)
		} else {
			int indIni=indiceDentro;
			do {
				//avanzamos por Tr ciclando (por si indiceDentro está por delante del coche)
				indiceDentro=(indiceDentro+1)%Tr.length;
			} while(indiceDentro!=indIni
					&& distanciaPuntos(posicionLocal,Tr[indiceDentro])>anchoCamino);
			if(indiceDentro==indIni) {
				System.out.println("estoy fuera ("+posicionLocal[0]+","+posicionLocal[1]+")");
				return dist;
			}
		}
		double[] v={Math.cos(Yaw), Math.sin(Yaw)};
		
		//Buscamos el índice del punto en mi barrido
		double AngIniB=barrAct.getAngulo(0);  //angulo inicial del barrido
		double AngFinB=barrAct.getAngulo(barrAct.numDatos()-1); //angulo final del barrido
		//por la derecha
		double[] v2=new double[2];
		boolean encontradoIniDer=false;
		for(iptoD=indiceDentro; iptoD<Bd.length; iptoD++) {
			if(distanciaPuntos(posicionLocal, Bd[iptoD])>barrAct.getDistanciaMaxima()) {
				System.out.println("Punto Decho muy lejos para considerarlo "+iptoD);
				break; //punto muy lejos para considerarlo
			}
			v2[0]=Bd[iptoD][0]-posicionLocal[0];	v2[1]=Bd[iptoD][1]-posicionLocal[1];
			double angD=anguloVectores(v, v2)+Math.PI/2;
			if (angD>=AngIniB && angD<=AngFinB) {
				encontradoIniDer=true;
				break; //este punto está dentro del barrido
			}
		}
		if(!encontradoIniDer) {
			System.out.println("Ningún punto de la derecha en el barrido");
			avanD=false; //no avanzamos por la derecha
			iptoD=indiceDentro;
		}
		iptoDini=iptoD; //apuntamos en indice inicial
			
		//Por la izquierda
		boolean encontradoIniIzd=false;
		for(iptoI=indiceDentro; iptoI<Bi.length; iptoI++) {
			if(distanciaPuntos(posicionLocal, Bi[iptoI])>barrAct.getDistanciaMaxima()) {
				System.out.println("Punto Izdo muy lejos para considerarlo "+iptoI);
				break; 
			}
			v2[0]=Bi[iptoI][0]-posicionLocal[0];	v2[1]=Bi[iptoI][1]-posicionLocal[1];
			double angI=anguloVectores(v, v2)+Math.PI/2;
			if (angI>=AngIniB && angI<=AngFinB) {
				encontradoIniIzd=true;
				break; //este punto está dentro del barrido
			}
		}
		if(!encontradoIniIzd) {
			System.out.println("Ningún punto de la izquierda en el barrido");
			avanI=false; //no avanzaremos por izquierda
			iptoI=indiceDentro;
		}
		iptoIini=iptoI; //apuntamos en indice inicial
				
		//procedemos a recorrer los puntos para ver si hay algo dentro
		iptoI--;  iptoD--; //se incrementan al entrar en el bucle
		while( avanD || avanI ) {
			if (avanD) {
				if(iptoD==(Bd.length-1)) {
					log("No queda borde derecho, no podemos seguir avanzando");
					avanD=false;
				} else {
					//avanzamos por la derecha
					iptoD++;
					double AngDant=AngD;
					double[] v2D={Bd[iptoD][0]-posicionLocal[0], Bd[iptoD][1]-posicionLocal[1]};
					double DistD=largoVector(v2D);
					AngD=anguloVectores(v, v2D)+Math.PI/2; //ya que 0º está 90º a la derecha
					if (AngD<AngDant) {
						log("El camino gira Decha Dejamos derecha");
						avanD=false;
					} else {
						//Vemos distancia barrido en angulo encontrado
						iAD=(int)Math.floor((AngD-AngIniB)/resAng);
						if(iAD>=barr.numDatos()) {
							log("No queda barrido para pto Decha");
							iAD=barr.numDatos()-1; 
							avanD=false;						
						} else if(barr.getDistancia(iAD)<barr.getDistanciaMaxima() 
								&& DistD>barr.getDistancia(iAD)) {
							ColDecha=true;
							avanD=false;						
						} else if (iAD<(barr.numDatos()-1) 
								&& barr.getDistancia(iAD+1)<barr.getDistanciaMaxima() 
								&& DistD>barr.getDistancia(iAD+1)) {
							iAD++;  
							ColDecha=true;
							avanD=false;
						}
					}
				}
			}
			if (avanI) {
				//avanzamos por la izda.
				if(iptoI==(Bi.length-1)) {
					log("No queda borde izdo, no podemos seguir avanzando"	);
					avanI=false;
				} else {
					iptoI++;
					double AngIant=AngI;
					double[] v2I={Bi[iptoI][0]-posicionLocal[0], Bi[iptoI][1]-posicionLocal[1]};
					double DistI=largoVector(v2I);
					AngI=anguloVectores(v, v2I)+Math.PI/2; //ya que 0º está 90º a la derecha
					if (AngI>AngIant) {
						log("El camino gira Izquierda. Dejamos izquierda");
						avanI=false;
					} else {
						//Vemos distancia barrido en angulo encontrado
						iAI=(int)Math.ceil((AngI-AngIniB)/resAng);
						if(iAI<0) {
							log("No queda barrido para pto Izda");
							iAI=0;
							avanI=false;
						} else if(barr.getDistancia(iAI)<barr.getDistanciaMaxima() 
								&& DistI>barr.getDistancia(iAI)) {
							ColIzda=true;
							avanI=false;						
						} else if ((iAI>0) 
								&& barr.getDistancia(iAI-1)<barr.getDistanciaMaxima() 
								&& DistI>barr.getDistancia(iAI-1)) {
							iAI--;
							ColIzda=true;
							avanI=false;
						}
					}
				}
			}
			
		} //fin del while

		//Buscamos segmento del coche
		indiceCoche=indiceDentro;
		while(!dentroSegmento(posicionLocal, indiceCoche)) { indiceCoche++; }
		indiceCoche++; //nos quedamos con el siguiente

		indSegObs=Integer.MAX_VALUE;
		indBarrSegObs=Integer.MAX_VALUE;
		if(iAD<iAI || (ColDecha && ColIzda)) {
			log("Los rayos no se han cruzado o hay 2 colisiones");
			if(!ColDecha && !ColIzda) {
				log("No se han cruzado y no hay colisión ??");
			}
			//buscamos posible minimo en medio
			indMin=iAD;
			for(int i=iAD+1; i<=iAI; i++)
				if(barr.getDistancia(i)<barr.getDistancia(indMin))
					indMin=i;
			dist=barr.getDistancia(indMin);
			//Tenemos que buscar pto dentro del camino más cercano
			indSegObs=Tr.length; //para limitar la búsquda
			indBarrSegObs=Integer.MAX_VALUE;
			for(int i=iAD; i<=iAI; i++) {
				double angI=barr.getAngulo(i);
				double distI=barr.getDistancia(i);
				double[] ptoI={posicionLocal[0]+distI*Math.cos(Yaw+angI-Math.PI/2)
						,posicionLocal[1]+distI*Math.sin(Yaw+angI-Math.PI/2)};
				//Buscamos segmento en que está
				//TODO usar algo más eficiente que la fuerza bruta
				int iSA=indiceCoche-1; //empezamos desde el coche
				while (iSA<indSegObs && !dentroSegmento(ptoI, iSA)) {
					//no está, vamos avanzando
					iSA++;
				} 
				if(dentroSegmento(ptoI, iSA)) {
					indSegObs=iSA; //se encontró uno más cercano
					indBarrSegObs=i;
				}
//				indSegObs=(ColDecha?iptoD:iptoI);
//				if(!dentroSegmento(pto, indSegObs)) {
//				//si ya está dentro empezamos acercándonos
//				do {
//				indSegObs--;
//				} while (indSegObs>indiceDentro && !dentroSegmento(pto, indSegObs));
//				if(indSegObs==indiceDentro) indSegObs=Integer.MAX_VALUE;
//				} 
//				else {
//				//no está, vamos avanzando
//				do { indSegObs++;} while (indSegObs<Tr.length && !dentroSegmento(pto, indSegObs));
//				if(!dentroSegmento(pto, indSegObs))
//				indSegObs=Integer.MAX_VALUE; //no se encontró
//				else
//				indSegObs--; // nos quedamos con el anterior
//				}
			}
			if(indSegObs==Tr.length) {
				log("No se encontró segmento ");
				//usamos hasta donde hemos podido explorar
				indSegObs=(iptoD<iptoI)?iptoI:iptoD;
				indBarrSegObs=Integer.MAX_VALUE;
			} //else indSegObs--; //nos quedamos con el anterior
		} else {
			log("los rayos se han cruzado y no hay colisión en los 2");
			if(ColIzda) {
				//usamos el punto de la trayectoria hasta donde podemos llegar
				dist=-distanciaPuntos(Tr[iLibre=iptoI],posActual);
				//Tenemos que buscar pto dentro del camino más cercano
				//TODO optimizar límites de búsqueda
				indSegObs=Tr.length; //para limitar la búsquda
				indBarrSegObs=Integer.MAX_VALUE;
				for(int i=iAI;i>=0; i--) {
					double angI=barr.getAngulo(i);
					double distI=barr.getDistancia(i);
					double[] ptoI={posicionLocal[0]+distI*Math.cos(Yaw+angI-Math.PI/2)
							,posicionLocal[1]+distI*Math.sin(Yaw+angI-Math.PI/2)};
					//Buscamos segmento en que está
					//TODO usar algo más eficiente que la fuerza bruta
					int iSA=indiceCoche-1; //empezamos desde el coche
					while (iSA<indSegObs && !dentroSegmento(ptoI, iSA)) {
						//no está, vamos avanzando
						iSA++;
					} 
					if(dentroSegmento(ptoI, iSA)) {
						indSegObs=iSA; //se encontró uno más cercano
						indBarrSegObs=i;
					}
				}
				if(indSegObs==Tr.length) {
					log("No se encontró segmento a persar colision");
					//usamos hasta donde hemos podido explorar
					indSegObs=iptoI;
					indBarrSegObs=Integer.MAX_VALUE;
				} //else indSegObs--; //nos quedamos con el anterior
			} else  if(ColDecha) {
				dist=-distanciaPuntos(Tr[iLibre=iptoD],posActual);
				//Tenemos que buscar pto dentro del camino más cercano
				//TODO optimizar límites de búsqueda
				indSegObs=Tr.length; //para limitar la búsquda
				indBarrSegObs=Integer.MAX_VALUE;
				for(int i=iAD;  i<barr.numDatos(); i++) {
					double angI=barr.getAngulo(i);
					double distI=barr.getDistancia(i);
					double[] ptoI={posicionLocal[0]+distI*Math.cos(Yaw+angI-Math.PI/2)
							,posicionLocal[1]+distI*Math.sin(Yaw+angI-Math.PI/2)};
					//Buscamos segmento en que está
					//TODO usar algo más eficiente que la fuerza bruta
					int iSA=indiceCoche-1; //empezamos desde el coche
					while (iSA<indSegObs && !dentroSegmento(ptoI, iSA)) {
						//no está, vamos avanzando
						iSA++;
					} 
					if(dentroSegmento(ptoI, iSA)) {
						indSegObs=iSA; //se encontró uno más cercano
						indBarrSegObs=i;
					}
				}
				if(indSegObs==Tr.length) {
					log("No se encontró segmento a persar colision");
					//usamos hasta donde hemos podido explorar
					indSegObs=iptoD;
					indBarrSegObs=Integer.MAX_VALUE;
				} //else indSegObs--; //nos quedamos con el anterior
			} else {
				//no ha colisionado ningún lado, cogemos el índice mayor
				//tambien para el camino
				iLibre=indSegObs=(iptoD<iptoI)?iptoI:iptoD;
				indBarrSegObs=Integer.MAX_VALUE;
				dist=-distanciaPuntos(Tr[iLibre],posActual);				
			}
		}
		log("Indice del segmento coche "+indiceCoche
				+" anterior al obstáculo "+indSegObs);
		distCamino=largoTramo(indiceCoche,indSegObs);
		return distCamino;
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
	 * Dice si pto pasado esta en cuadrilátero de la trayectoria
	 * @param pto por el que se pregunt
	 * @param i cuadrilátero i-ésimo del camino
	 * @return si está dentro
	 */
	public boolean dentroSegmento(double[] pto,int i){
		if(i>=(Tr.length-1))
			return false;
		double sumAng=0;
		double[] vA={pto[0]-Bi[i][0], pto[1]-Bi[i][1]};
		double[] vB={pto[0]-Bi[i+1][0], pto[1]-Bi[i+1][1]};
		double[] vC={pto[0]-Bd[i+1][0], pto[1]-Bd[i+1][1]};
		double[] vD={pto[0]-Bd[i][0], pto[1]-Bd[i][1]};
		
		log("esq=["+Bi[i][0]+","+Bi[i][1]+";"
				+Bi[i][0]+","+Bi[i][1]+";"
				+Bi[i+1][0]+","+Bi[i+1][1]+";"
				+Bd[i+1][0]+","+Bd[i+1][1]+";"
				+Bd[i][0]+","+Bd[i][1]+";"
				+"], pto=["+pto[0]+","+pto[1]+"]"
				);
		
		
		sumAng+=anguloVectores(vA, vB);
		sumAng+=anguloVectores(vB, vC);
		sumAng+=anguloVectores(vC, vD);
		sumAng+=anguloVectores(vD, vA);
		
		return (Math.abs(Math.abs(sumAng)-(2*Math.PI))<1e-3);

	}
	
	
	public double largoTramo(int iini, int ifin) {
		if(iini>ifin || ifin>=Tr.length)
			return Double.POSITIVE_INFINITY;
		double largo=0;
		for(int i=iini; i<ifin; i++) {
			double[] v={Tr[i+1][0]-Tr[i][0], Tr[i+1][1]-Tr[i][1]}; 
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
		ret+="\n iAD="+iAD
				+" iAI="+iAI
				+" ColDecha =" + ColDecha
				+" ColIzda ="+ ColIzda
				+"\n iptoD ="+iptoD
				+"  iptoI ="+iptoI
				+"  iptoDini ="+iptoDini
				+"  iptoIini ="+iptoIini
				+" \n imin ="+indMin
				+" \n indiceCoche ="+indiceCoche
				+" indSegObs ="+indSegObs
				+" indBarrSegObs ="+indBarrSegObs
		;
		return ret;
	}
}
