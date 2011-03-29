package boids;

import sibtra.gps.Trayectoria;

import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.io.File;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
import java.util.Random;
import javax.swing.JFileChooser;

import sibtra.lms.BarridoAngular;
import sibtra.predictivo.*;
import sibtra.util.*;
import sibtra.log.LoggerArrayDoubles;
import sibtra.log.LoggerFactory;
//import predictivo.UtilCalculos;

import Jama.Matrix;

public class Simulador {
	double tiempoAnt;
	public double getTiempoAnt() {
		return tiempoAnt;
	}

	public void setTiempoAnt(double tiempoAnt) {
		this.tiempoAnt = tiempoAnt;
	}

	
	public Coche getModCoche() {
		return modCoche;
	}

	public void setModCoche(Coche modCoche) {
		this.modCoche = modCoche;
	}

	Vector<Matrix> rutaDinamica = new Vector<Matrix>();
	Trayectoria tr;
	Coche modCoche = new Coche();
	int horPrediccion = 13;
	int horControl = 3;
	double landa = 1;
	double Ts = 0.05;
	ControlPredictivo contPred = new ControlPredictivo(modCoche,tr, horPrediccion, horControl, landa, Ts);	
	JFileChooser selectorArchivo = new JFileChooser(new File("./Simulaciones"));
	/** Coordenadas a partir de las cuales se situa la bandada*/
	Matrix posInicial = new Matrix(2,1);
	/** Coordenadas del objetivo que han de alcanzar los boids*/
	Matrix objetivo = new Matrix(2,1);
	/** Máximo valor en segundos que se permite para cada simulación*/
	double tiempoMax;
	/** Tiempo invertido en que el numBoidsOk lleguen al objetivo*/
	double tiempoInvertido;
	/** Número de boids que han alcanzado el objetivo*/
	int numBoidsOk;
	/** Número de boids que han alcanzado el objetivo y se consideran suficientes para detener
	 *  la simulación*/
	int numBoidsOkDeseados;
	/** Número de iteraciones que tarda la simulación en alcanzar la condición de éxito*/
	int contIteraciones = 0;
	/** Distancia a la que se considera que se ha alcanzado el objetivo*/
	double distOk;
	/** Número de boids que forman la bandada*/
	int tamanoBandada;
	/** Vector donde se alamacenan los boids de la bandada*/
	Vector<Boid> bandada = new Vector<Boid>();
	/** Vector donde se almacenan los boids que han alcanzado el objetivo*/
	Vector<Boid> boidsOk = new Vector<Boid>();
	/** Vector con la información de posición de los obstáculos del escenario*/
	Vector<Obstaculo> obstaculos = new Vector<Obstaculo>();
	/** Vector que contiene los puntos de diseño para la simulación por lotes*/
	Vector <Hashtable> vectorSim = new Vector<Hashtable>();
	Vector <TipoCamino> caminos = new Vector<TipoCamino>();
	private int incrNuevosBoids = 2;
	private int incrPensar = 1;
	private int contNuevosBoids = incrNuevosBoids ;
	private int contPensar = 0;
	private double comando = 0;
	public double anchoEscenario = 0;
	public double largoEscenario = 0;
	private LoggerArrayDoubles logPosturaCoche;
	private double distOkAlOrigen = 5;
	private boolean rutaCompleta;
	
	
	//-------------Constructores---------------------------------------------------
	

	public Vector<TipoCamino> getCaminos() {
		return caminos;
	}

	public void setCaminos(Vector<TipoCamino> caminos) {
		this.caminos = caminos;
	}

	/**Constructor por defecto*/
	public Simulador(){
		setTamanoBandada(20);
		crearBandada(getTamanoBandada());
		posicionarBandada(new Matrix(2,1));
		setObjetivo(new Matrix(2,1));
		setTiempoMax(5);
		setDistOk(2);
		setNumBoidsOkDeseados(2);
		logPosturaCoche=LoggerFactory.nuevoLoggerArrayDoubles(this, "PosturaCoche");
		logPosturaCoche.setDescripcion("Coordenadas y yaw [x,y,yaw]");
	}
	
	public Simulador(Matrix puntoIni,Matrix objetivo,double tMax,int boidsOk,
			         double distanOk,int sizeBandada,Vector<Obstaculo> obstaculos){
		
		setTamanoBandada(sizeBandada);
		crearBandada(getTamanoBandada());
		posicionarBandada(puntoIni);
		setObstaculos(obstaculos);
		setObjetivo(objetivo);
		setTiempoMax(tMax);
		setDistOk(distanOk);
		setNumBoidsOk(boidsOk);
		
	}
	
	//-------------Métodos para el manejo de la bandada----------------------------
	
	/**
	 *  Crea una bandada de Boids
	 * @param numBoids cantidad de boids deseado para la bandada
	 */
	public void crearBandada(int numBoids){
		if (getBandada().size() > 0)
			borrarBandada();
		setTamanoBandada(numBoids);
    	for (int i=0; i< tamanoBandada;i++){
    		getBandada().add(new Boid(new Matrix(2,1),new Matrix(2,1),new Matrix(2,1)));	
    	}
    	posicionarBandada(posInicial);
	}
	public void crearBandada(){
//		if (getBandada().size() > 0)
//			borrarBandada();
		getBandada().clear();
    	for (int i=0; i< tamanoBandada;i++){
    		getBandada().add(new Boid(new Matrix(2,1),new Matrix(2,1),new Matrix(2,1)));	
    	}
    	posicionarBandada(posInicial);
	}
	
	/**Limpia el vector de boids*/	
	public void borrarBandada(){
		bandada.clear();
//		setTamanoBandada(0);
	}
	/**
	 * Posiciona la bandada en un punto
	 * @param puntoIni Matriz 2x1 que indica el punto alrededor del cual se va a colocar la 
	 * bandada
	 */
	public void posicionarBandada(Matrix puntoIni){
		if (getBandada().size()>0){
			for (int i=0;i<getBandada().size();i++){				
//				double pos[] = {e.getX()+Math.random()*getTamanoBan()*2, e.getY()+Math.random()*getTamanoBan()*2};
				double pos[] = {puntoIni.get(0,0)+Math.random(), puntoIni.get(1,0)+Math.random()};
				Matrix posi = new Matrix(pos,2);
				double vel[] = {Math.random(),Math.random()};
				Matrix velo = new Matrix(vel,2);
				this.getBandada().elementAt(i).resetRuta();
//				this.getBandada().elementAt(i).getForma().transform(AffineTransform.getTranslateInstance(pos[0]-getBandada().elementAt(i).getPosicion().get(0,0),
//						pos[1]-getBandada().elementAt(i).getPosicion().get(1,0)));
				this.getBandada().elementAt(i).setPosicion(posi);			
				this.getBandada().elementAt(i).setVelocidad(velo);
				double ace[] = {0,0};
				Matrix acel = new Matrix(ace,2);
				this.getBandada().elementAt(i).setAceleracion(acel);
			}
		}
	}
	
	/**
	 * Retira un boid de la bandada principal y lo inserta en la bandada de los boids que han
	 * alcanzado el objetivo
	 * @param indBoid indice del Boid que se desea trasladar
	 */
	public void traspasarBoid(int indBoid){
		if(bandada.size()>0){
			boidsOk.add(bandada.remove(indBoid));
//			setTamanoBandada(bandada.size());
		}
		else
			System.err.println("La bandada principal está vacía");
	}
	//-----------Fin de los métodos para manejar la bandada-----------------
	
	//-----------Métodos para la simulación---------------------------------
	/**
	 * Método que genera tantos obstáculos como numObst, con una magnitud de velocidad velMax
	 * con dirección aleatoria. La posición de los obstáculos también es aleatoria
	 */
	public void generaObstaculos(int numObst,double velMax){
		//Se eliminan los obstáculos que pudiera haber de anteriores simulaciones
		obstaculos.clear();
		Random rand = new Random();
		for (int i=0;i<numObst;i++){
			double posX = Math.random()*largoEscenario;
			double posY = Math.random()*anchoEscenario;
			double velX = rand.nextGaussian()*velMax;
			double velY = rand.nextGaussian()*velMax;
			double rumboX = velX;
			double rumboY = velY;
			Obstaculo obs = new Obstaculo(posX, posY, velX, velY, rumboX, rumboY);
			obstaculos.add(obs);
		}
	}
	
	
	public double calculaComandoVolante(){
		double comando = 0;
		double Kp = 0.5;
		double cotaAngulo=Math.toRadians(30);
		Matrix ultimoPunto = new Matrix(2,1);
		if (rutaDinamica.size() <= 1){
			System.out.println("La ruta dinámica está vacia");
		}else{
			ultimoPunto.set(0,0, rutaDinamica.elementAt(1).get(0,0));
			ultimoPunto.set(1,0, rutaDinamica.elementAt(1).get(1,0));
//			ultimoPunto.set(0,0, rutaDinamica.lastElement().get(0,0));
//			ultimoPunto.set(1,0, rutaDinamica.lastElement().get(1,0));
//			ultimoPunto.set(0,0, 0);
//			ultimoPunto.set(1,0, 0);
		}
		Matrix vectorDif = ultimoPunto.minus(posInicial);
//		Matrix vectorDif = objetivo.minus(posInicial);
//		double difAngular = UtilCalculos.normalizaAngulo(Math.atan2(vector.get(1,0),vector.get(0,0)));
		double AngAlPunto = Math.atan2(vectorDif.get(1,0),vectorDif.get(0,0));
		double difAngular = UtilCalculos.normalizaAngulo(AngAlPunto - modCoche.getYaw());
		comando = UtilCalculos.limita(difAngular*Kp,-cotaAngulo,cotaAngulo);
//		comando = UtilCalculos.limita(contPred.calculaComando(),-cotaAngulo,cotaAngulo);
//		System.out.println("acabó el control predictivo");
//		System.out.println("el comando es " + comando);
//		System.out.println("La orient del coche "+modCoche.getYaw()+"La difAng es "+difAngular+" y el comando es "+comando);
		return comando;
	}
	
	public double calculaVelocidadCoche(){
		double velocidad = 0;
		double paradaEmergencia = 1;
		double vel[] = {Math.cos(modCoche.getYaw()),Math.sin(modCoche.getYaw())};
		Matrix velo = new Matrix(vel,2);
		for (int i=0;i<getObstaculos().size();i++)
		paradaEmergencia = calculaParadaEmergencia(getObstaculos().elementAt(i).getPosicion(),
				posInicial,velo);
		if (!isRutaCompleta()){
			velocidad = -1*paradaEmergencia;
		}else{
			velocidad = 3*paradaEmergencia;
		}
		return velocidad;
	}
	
	public void moverPtoInicial(double tiempoActual,double Ts){

//		if(Math.abs(tiempoActual-tiempoAnt) > 500){
			double velocidad = calculaVelocidadCoche();
//			if (tr.length()>3)
				comando  = calculaComandoVolante();
			
//			contPred.setTs(Ts/1000);
//			comando  = contPred.calculaComando();
//			comando = sibtra.util.UtilCalculos.limita(comando,-Math.PI/6,Math.PI/6);
//			System.out.println("el comando calculado es " + comando);
			setTiempoAnt(tiempoActual);
//		}   
			modCoche.calculaEvolucion(comando,velocidad,Ts);
			posInicial.set(0,0,modCoche.getX());
			posInicial.set(1,0,modCoche.getY());
			logPosturaCoche.add(modCoche.getX(),modCoche.getY(),modCoche.getYaw());
//			System.out.println("yaw del coche "+modCoche.getYaw());
	}
	/**
	 * Calcula el desplazamiento y mueve cada uno de los Boids de la bandada. Se le pasa
	 * el índice del lider de la iteración anterior
	 */
	public int moverBoids(int indMinAnt){
		int indLider = 0;
		double distMin = Double.POSITIVE_INFINITY;
		boolean liderEncontrado = false;
		contIteraciones++;
		//If para controlar la frecuencia a la que se añaden boids a la bandada
		if(contIteraciones > contNuevosBoids){
			for(int g=0;g<2;g++){				
//				double pos[] = {Math.abs(700*Math.random()),Math.abs(500*Math.random())};
//				double pos[] = {getBandada().lastElement().getPosicion().get(0,0)+10*Math.random(),
//						getBandada().lastElement().getPosicion().get(1,0)+10*Math.random()};
				double pos[] = {posInicial.get(0,0)+Math.random(), posInicial.get(1,0)+Math.random()};
				Matrix posi = new Matrix(pos,2);
				double vel[] = {0,0};
				Matrix velo = new Matrix(vel,2);
				double ace[] = {0,0};
				Matrix acel = new Matrix(ace,2);
				getBandada().add(new Boid(posi,velo,acel));
			}
//			double posCentroEscenario[] = {largoEscenario/2,anchoEscenario/2};
//			Matrix posiCentro = new Matrix(posCentroEscenario,2);
//			double vel[] = {0,0};
//			Matrix velo = new Matrix(vel,2);
//			double ace[] = {0,0};
//			Matrix acel = new Matrix(ace,2);
//			getBandada().add(new Boid(posiCentro, velo, acel));
			contNuevosBoids = contIteraciones + incrNuevosBoids;
		}
		// Iteramos sobre toda la bandada		
		if (getBandada().size() != 0){
//			System.out.println("Tamaño actual de la bandada " + getBandada().size());
		
			for (int j = 0;j<getBandada().size();j++){
				getBandada().elementAt(j).setConectado(false);
				getBandada().elementAt(j).calculaValoracion();
				getBandada().elementAt(j).setAntiguo(contIteraciones);
				if(contIteraciones > contPensar){
					getBandada().elementAt(j).calculaMover(getBandada()
						,getObstaculos(),j,Boid.getObjetivo());					
				}
//				getBandada().elementAt(j).mover(getBandada()
//						,getObstaculos(),j,Boid.getObjetivo());
				getBandada().elementAt(j).mover();
				double dist = getBandada().elementAt(j).getDistObjetivo();
				// Deshabilitamos el liderazgo de la iteración anterior antes de retirar ningún 
				// de la bandada por cercanía al objetivo				
//				Si está lo suficientemente cerca del objetivo lo quitamos de la bandada
				if (dist < distOk){
					getBandada().elementAt(j).setNumIteraciones(getContIteraciones());
					traspasarBoid(j);
					numBoidsOk++; // Incremento el numero de boids que han llegado al objetivo
				}
				// Buscamos al lider
				if(j < getBandada().size()){
					if (getBandada().elementAt(j).isCaminoLibre()){										
						if (dist < distMin){
							distMin = dist;
							indLider = j;
							liderEncontrado = true;
						}
					}
				}
									
			}
			if (contIteraciones > contPensar){
				contPensar = contIteraciones + incrPensar;			
			}
			
			if (indMinAnt<getBandada().size())
				getBandada().elementAt(indMinAnt).setLider(false);
			if (liderEncontrado && (indLider<getBandada().size())){
				getBandada().elementAt(indLider).setLider(true);
			}
		}
		return indLider;				
	}
	/**
	 * Método que utiliza la información de un barrido de rangeFinder para posicionar 
	 * los obstáculos en el escenario
	 * @param ba Barrido de un rangeFinder, tanto LMS221 como LMS112
	 */
	public Vector<Obstaculo> posicionarObstaculos(BarridoAngular ba){
		getObstaculos().clear(); // Se eliminan los obstáculos que pudieran existir en el
								// escenario
		for (int i = 0; i<ba.numDatos();i++){
			double ang=ba.getAngulo(i);
			double dis=ba.getDistancia(i)*500;
//			System.out.println("distancia medida " + dis);
			double pos[] = {Math.abs(dis*Math.cos(ang)),Math.abs(dis*Math.sin(ang))};
			double vel[] = {0,0}; // En un futuro tal vez se tenga una estimación de la velocidad
			getObstaculos().add(new Obstaculo(new Matrix(pos,2),new Matrix(vel,2)));
//			getObstaculos().get(i).posicion.print(2,2);
			
		}
		return getObstaculos();
		//Usando un BarridoAngularIterator en lugar de BarridoAngular
//		while(ba.next()){
//			double ang=ba.angulo();
//			double dis=ba.distancia();
//			double pos[] = {dis*Math.cos(ang),dis*Math.sin(ang)};
//			double vel[] = {0,0}; // En un futuro tal vez se tenga una estimación de la velocidad
//			getObstaculos().add(new Obstaculo(new Matrix(pos,2),new Matrix(vel,2)));
//		}
	}
	
	/**
	 * Recorre el vector de obstáculos y dependiendo de los valores de velocidad y aceleración 
	 * de cada obstáculo se calcula el desplazamiento y se actualiza el valor de posición de 
	 * cada obstáculo
	 */
	public void moverObstaculos(){
		if(getObstaculos().size() != 0){
			for(int i = 0;i<getObstaculos().size();i++){	
				double gananciaVel = calculaParadaEmergencia(posInicial,
						getObstaculos().elementAt(i).getPosicion(),
						getObstaculos().elementAt(i).getVelocidad());
				getObstaculos().elementAt(i).mover(
						getObstaculos().elementAt(i).getRumboDeseado().times(gananciaVel),Ts);
				//-Control para que los obstáculos vuelvan a aparecer por el lado contrario-
				//-al salirse del escenario
				if(getObstaculos().elementAt(i).getPosicion().get(1,0)>anchoEscenario){
					double pos[] = {getObstaculos().elementAt(i).getPosicion().get(0,0),0};
					Matrix posi = new Matrix(pos,2);
					getObstaculos().elementAt(i).setPosicion(posi);
				}
				if(getObstaculos().elementAt(i).getPosicion().get(1,0)<0){
					double pos[] = {getObstaculos().elementAt(i).getPosicion().get(0,0),anchoEscenario};
					Matrix posi = new Matrix(pos,2);
					getObstaculos().elementAt(i).setPosicion(posi);
				}

				if(getObstaculos().elementAt(i).getPosicion().get(0,0)>largoEscenario){
					double pos[] = {0,getObstaculos().elementAt(i).getPosicion().get(1,0)};
					Matrix posi = new Matrix(pos,2);
					getObstaculos().elementAt(i).setPosicion(posi);
				}	
			}
		}
	}
	/**
	 * Dependiendo de la posición del vehículo el obstáculo se detendrá o no
	 * @param Obs Obstáculo sobre el que se quiere calcular su velocidad 
	 * @return velocidad
	 */
	public double calculaParadaEmergencia(Matrix posObst,Matrix posObjMovil, Matrix velObjMovil){
		double veloObjMovil = 0;		
		// calculo el vector diferencia entre la pos del obstáculo
		// y la posicion inicial de salida de los boids, es decir, la posición del
		// vehículo
		Matrix difPosInicialObs = new Matrix(2,1);
		difPosInicialObs = posObst.minus(posObjMovil);
		double dist = difPosInicialObs.norm2();
		// calculo el ángulo entre la velocidad del obstáculo y el vector diferencia 
		// anteriormente calculado para saber si el coche está dentro de la trayectoria
		// del obstáculo
		 // OJO CON EL MENOS DE LA COMPONENTE Y, EN ES SIST DE REF DE LA PANTALLA EL EJE Y
		 // ESTÁ INVERTIDO!!
		double orientacionObjMovil = Math.atan2(velObjMovil.get(1,0)
				,velObjMovil.get(0,0));
		double orientacionVectorDif = Math.atan2(difPosInicialObs.get(1,0)
				,difPosInicialObs.get(0,0));		
		double difOrientaciones = UtilCalculos.normalizaAngulo(orientacionObjMovil-orientacionVectorDif);

		if ((Math.abs(difOrientaciones) < Math.PI/6)&&(dist < 3)){
			veloObjMovil = 0;
		}
		else{
			veloObjMovil = 1;
		}
		return veloObjMovil;
	}

	public void simuPorLotes(){
		int indMinAnt = 0;
		double tiempoIni = System.currentTimeMillis();
		tiempoInvertido = 0;
		setNumBoidsOk(0);
		contIteraciones = 0;
//		int contNuevosBoids = 20;
//		int ind = 0;
//			 Bucle while que realiza una simulación completa, es decir, hasta que lleguen
			// los boids especificados o hasta que se cumpla el tiempo máximo
		while ((tiempoInvertido < tiempoMax) && (numBoidsOk < numBoidsOkDeseados)){
			indMinAnt =  moverBoids(indMinAnt);
			tiempoInvertido = (System.currentTimeMillis()-tiempoIni)/1000;
			contIteraciones++; // Llevamos la cuenta de las iteraciones del bucle principal de 
			// la simulación
//			bandada.add(new Boid(posInicial,new Matrix(2,1)));
//			getBandada().remove(ind);
//			System.out.println("tamaño de la bandada " + bandada.size());
//			if(ind < bandada.size()){
//				ind++;
//			}
//			if(contIteraciones > contNuevosBoids){
//				
//				contNuevosBoids = contNuevosBoids + 20;
//			}
			
		}
		// Escribimos los datos en un fichero
//		int devuelto = selectorArchivo.showSaveDialog(null);
//        if (devuelto == JFileChooser.APPROVE_OPTION) {
//            File fichero = selectorArchivo.getSelectedFile();
//            try {
//    			File file = new File(fichero.getAbsolutePath());
//    			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file));
//    			oos.writeObject(this);
//    			oos.close();
//    		} catch (IOException ioe) {
//    			System.err.println("Error al escribir en el fichero ");
//    			System.err.println(ioe.getMessage());
//    		}
//        }        
	}
	/**
	 * Método que se encarga de crear la trayectoria hasta el objetivo usando la técnica de
	 * la cadena de boids
	 * @param indLider Se le pasa el índice del boid más cercano y con visión directa hasta el objetivo
	 * @return
	 */
	
	public Vector<Matrix> calculaRutaDinamica(int indLider){	
		rutaDinamica.clear();
		setRutaCompleta(false);
		int boidActual = indLider;
		int boidAux = 0;
		int cont = 0;
		boolean encontrado = false;
		double valoracion = Double.NEGATIVE_INFINITY;
		double umbralCercania = 5; 
//		System.out.println("Empezó nueva ruta");
		rutaDinamica.add(posInicial);
		while (cont < getBandada().size()){
			cont++;
			encontrado=false;
			for (int i=0;i < bandada.size();i++){				
				if (i != boidActual){// No se comprueba consigo mismo
					if(!getBandada().elementAt(i).isConectado()){// El boid elegido no puede estar
						// conectado con otro
						double dist = bandada.elementAt(boidActual).getPosicion().minus(getBandada().elementAt(i).getPosicion()).norm2();
						if (dist < umbralCercania){// Tiene que estar lo suficientemente cerca
							boolean caminoOcupado = false;
							// Calculamos la recta entre ambos boids
							Line2D recta = 
								new Line2D.Double(getBandada().elementAt(boidActual).getPosicion().get(0,0),
										getBandada().elementAt(boidActual).getPosicion().get(1,0),
										getBandada().elementAt(i).getPosicion().get(0,0),
										getBandada().elementAt(i).getPosicion().get(1,0));
							for (int j=0;j < obstaculos.size();j++){
								double distObs = obstaculos.elementAt(j).getPosicion().minus(
										getBandada().elementAt(boidActual).getPosicion()).norm2();
								if (distObs < umbralCercania){
									if (!caminoOcupado)// Sólo se calcula la intersección mientras el camino siga sin ocupar
										caminoOcupado = recta.intersects(obstaculos.elementAt(j).getForma());
								}							
							}
							if(!caminoOcupado){
								if(getBandada().elementAt(i).getValoracion() > valoracion){
									boidAux = i;
									valoracion = getBandada().elementAt(i).getValoracion();
									encontrado = true;
								}
//								System.out.println("encontró compañero");
							}
						}
					}
				}
			}
			if(encontrado){
				getBandada().elementAt(boidAux).setConectado(true);
//				getBandada().elementAt(boidAux).setExperiencia(1);
//				System.out.println("La valoracion es : " + valoracion);
//				rutaDinamica.add(getBandada().elementAt(boidAux).getPosicion());
				rutaDinamica.add(getBandada().elementAt(boidAux).calculaCentroMasas(getBandada(),4));
				boidActual = boidAux;
//				System.out.println("saltó al siguiente boid");
			}
			
		}
//		rutaDinamica = mejoraRuta(rutaDinamica);
//		boolean caminoOcupadoPosIni = false;
//		// Calculamos la recta entre el ultimo punto de la ruta dinamica y la posición final
//		Line2D recta = 
//			new Line2D.Double(rutaDinamica.lastElement().get(0,0),
//					rutaDinamica.lastElement().get(1,0),
//					objetivo.get(0,0),objetivo.get(1,0));
//		for (int j=0;j < obstaculos.size();j++){// Se comprueba con todos los obstáculos
////			double distObs = obstaculos.elementAt(j).getPosicion().minus(
////					rutaDinamica.lastElement()).norm2();
////			if (distObs < umbralCercania){
//				if (!caminoOcupadoPosIni)// Sólo se calcula la intersección mientras el camino siga sin ocupar
//					caminoOcupadoPosIni = recta.intersects(obstaculos.elementAt(j).getForma());
////			}							
//		}
//		if(!caminoOcupadoPosIni){
//			rutaDinamica.add(objetivo);
//			setRutaCompleta(true);
//		}		
//		System.out.println("acabó la ruta");
		if (rutaDinamica.size()>=2){
			setRutaCompleta(true);
			tr = new Trayectoria(traduceRuta(rutaDinamica));
//			tr = new Trayectoria(tr,0.1);
			tr.situaCoche(posInicial.get(0,0),posInicial.get(1,0));
			contPred.setRuta(tr);
		}
//		rutaDinamica = mejoraRuta(rutaDinamica);
		return rutaDinamica;
	}
	/**
	 * Método que simplifica la ruta calculada por calculaRutaDinamica
	 * @param ruta
	 * @return
	 */
	
	public  Vector<Matrix> mejoraRuta(Vector<Matrix> ruta){
		Vector<Matrix> rutaMejor = new Vector<Matrix>();		
		int ptoBase=0;
		boolean caminoOcupado = false;
		rutaMejor.add(ruta.elementAt(ptoBase));
		for(int i=1;i<ruta.size();i++){
			Line2D recta = 
				new Line2D.Double(ruta.elementAt(ptoBase).get(0,0),ruta.elementAt(ptoBase).get(1,0)
						,ruta.elementAt(i).get(0,0),ruta.elementAt(i).get(1,0));
			for(int j=0;j<obstaculos.size() && !caminoOcupado;j++){				
				caminoOcupado = recta.intersects(obstaculos.elementAt(j).getForma());				
			}
			if (caminoOcupado){
				rutaMejor.add(ruta.elementAt(i-1));
				ptoBase = i-1;
				caminoOcupado = false;
			}
		}
		rutaMejor.add(ruta.elementAt(ruta.size()-1));		
		return rutaMejor;
	}
	/**
	 * Método que traduce la información de la ruta (vector de matrices) a un array de varias dimensiones
	 * @param ruta
	 * @return
	 */
	public double[][] traduceRuta(Vector<Matrix> ruta){
		if (ruta.size()==0){
			throw new IllegalArgumentException("la ruta tiene que tener puntos para poderla traducir");
		}
		double [][] arrayRuta = new double[ruta.size()][2];
		for (int i=0;i==ruta.size();i++){
			arrayRuta[i][0] = ruta.get(i).get(0,0);
			arrayRuta[i][1] = ruta.get(i).get(1,0);
		}
		return arrayRuta;
	}
	
	public void configurador(Hashtable designPoint,String[] nomParam){
		for (Enumeration e = designPoint.keys() ; e.hasMoreElements() ;) {
//	         System.out.println();
	         String param = (String)e.nextElement();
	         int indice = 0;
	         for(int i=0;i<nomParam.length;i++){  //Buscamos coincidencia en las etiquetas
	        	 if (param.equalsIgnoreCase(nomParam[i])){
	        		 indice = i;
	        		 break;
	        	 }	        	 
	         }
	         switch (indice){//Dependiendo de la etiqueta se varía un parámetro u otro
	         case 0: Boid.setRadioObstaculo((Double)designPoint.get(nomParam[indice]));break;
	         case 1: Boid.setRadioCohesion((Double)designPoint.get(nomParam[indice]));break;
	         case 2: Boid.setRadioSeparacion((Double)designPoint.get(nomParam[indice]));break;
	         case 3: Boid.setRadioAlineacion((Double)designPoint.get(nomParam[indice]));break;
	         case 4: Boid.setPesoCohesion((Double)designPoint.get(nomParam[indice]));break;
	         case 5: Boid.setPesoSeparacion((Double)designPoint.get(nomParam[indice]));break;
	         case 6: Boid.setPesoAlineacion((Double)designPoint.get(nomParam[indice]));break;
	         case 7: Boid.setPesoObjetivo((Double)designPoint.get(nomParam[indice]));break;
	         case 8: Boid.setPesoObstaculo((Double)designPoint.get(nomParam[indice]));break;
	         case 9: Boid.setPesoLider((Double)designPoint.get(nomParam[indice]));break;	         
	         case 10: Boid.setVelMax((Double)designPoint.get(nomParam[indice]));break;	         
	         case 11: setTamanoBandada((Double)designPoint.get(nomParam[indice]));
	         		  crearBandada();break;	         		  
	         case 12:setNumBoidsOkDeseados((Double)designPoint.get(nomParam[indice]));break;	         	         		  
	         }
	     }
	}
	
	//-----------------------------------------------------------------------
	//-------------------Getters y Setters-----------------------------------
	//-----------------------------------------------------------------------
	
//	public ControlPredictivo getCp() {
//		return contPred;
//	}
//
//	public void setCp(ControlPredictivo cp) {
//		this.contPred = cp;
//	}

	public Vector<Boid> getBandada() {
		return bandada;
	}

	public void setBandada(Vector<Boid> bandada) {
		this.bandada = bandada;
	}

	public int getNumBoidsOk() {
		return numBoidsOk;
	}

	public void setNumBoidsOk(int numBoidsOk) {
		this.numBoidsOk = numBoidsOk;
	}
	
	public void setNumBoidsOk(double numBoidsOk) {
		this.numBoidsOk = (int)numBoidsOk;
	}

	public Matrix getObjetivo() {
		return objetivo;
	}
	
	public int getContIteraciones() {
		return contIteraciones;
	}

	public void setObjetivo(Matrix objetivo) {
		this.objetivo = objetivo;
		Boid.setObjetivo(this.objetivo.get(0,0),this.objetivo.get(1,0));
	}
	
	public void setObjetivo(double Xobj,double Yobj) {
		double pos[] = {Xobj,Yobj};
		Matrix posi = new Matrix(pos,2);
		this.objetivo = posi;
		Boid.setObjetivo(this.objetivo.get(0,0),this.objetivo.get(1,0));
	}

	public Vector<Obstaculo> getObstaculos() {
		return obstaculos;
	}

	public void setObstaculos(Vector<Obstaculo> obstaculos) {
		this.obstaculos = obstaculos;
	}

	public Matrix getPosInicial() {
		return posInicial;
	}

	public void setPosInicial(Matrix posInicial) {
		this.posInicial = posInicial;
		this.modCoche.setPostura(posInicial.get(0,0),posInicial.get(1,0),0);
		posicionarBandada(posInicial);
		Boid.setPosInicial(posInicial);
	}

	public int getTamanoBandada() {
		return tamanoBandada;
	}

	public void setTamanoBandada(int tamanoBandada) {
		this.tamanoBandada = tamanoBandada;
	}
	public void setTamanoBandada(double tamanoBandada) {
		this.tamanoBandada = (int)tamanoBandada;
	}

	public double getTiempoMax() {
		return tiempoMax;
	}

	public void setTiempoMax(double tiempoMax) {
		this.tiempoMax = tiempoMax;
	}
	public Vector<Boid> getBoidsOk() {
		return boidsOk;
	}

	public void setBoidsOk(Vector<Boid> boidsOk) {
		this.boidsOk = boidsOk;
	}

	public double getDistOk() {
		return distOk;
	}

	public void setDistOk(double distOk) {
		this.distOk = distOk;
	}
	public int getNumBoidsOkDeseados() {
		return numBoidsOkDeseados;
	}

	public void setNumBoidsOkDeseados(int numBoidsOkDeseados) {
		this.numBoidsOkDeseados = numBoidsOkDeseados;
	}
	
	public void setNumBoidsOkDeseados(double numBoidsOkDeseados) {
		this.numBoidsOkDeseados = (int)numBoidsOkDeseados;
	}
	
	public double getTiempoInvertido() {
		return tiempoInvertido;
	}
	
	public double getAnchoEscenario() {
		return anchoEscenario;
	}

	public void setAnchoEscenario(double anchoEscenario) {
		this.anchoEscenario = anchoEscenario;
	}

	public double getLargoEscenario() {
		return largoEscenario;
	}

	public void setLargoEscenario(double largoEscenario) {
		this.largoEscenario = largoEscenario;
	}

	public double getTs() {
		return Ts;
	}

	public void setTs(double ts) {
		Ts = ts;
	}
	
	public boolean isRutaCompleta() {
		return rutaCompleta;
	}

	public void setRutaCompleta(boolean rutaCompleta) {
		this.rutaCompleta = rutaCompleta;
	}
	
	//----------------------------------------------------------------------------
	//----------------Final de los Getters y Setters------------------------------
	//----------------------------------------------------------------------------
}
