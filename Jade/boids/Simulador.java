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
import flanagan.*;
import flanagan.interpolation.CubicSpline;
import gridBasedSearch.Grid;
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
	Vector<Matrix> rutaDinamicaSuave = new Vector<Matrix>();

	Trayectoria tr;
	Coche modCoche = new Coche();
	Coche cocheSolitario = new Coche();
	int horPrediccion = 13;
	int horControl = 2;//3;
	double landa = 1;//1;
	double Ts = 0.1;
	double TsPred = 0.2;
	ControlPredictivo contPred = new ControlPredictivo(modCoche,tr, horPrediccion, horControl, landa, TsPred);	

	JFileChooser selectorArchivo = new JFileChooser(new File("./Simulaciones"));
	/**Coordenadas del coche con comportamiento reactivo*/
	Matrix posCocheSolitario = new Matrix(2,1);
	/**Vector velocidad del coche con comportamiento reactivo*/
	Matrix velCocheSolitario = new Matrix(2,1);
	/** Coordenadas a partir de las cuales se situa la bandada*/
	Matrix posInicial = new Matrix(2,1);
	/** Coordenadas del objetivo que han de alcanzar los boids*/
	Matrix objetivo = new Matrix(2,1);
	/** MÃ¡ximo valor en segundos que se permite para cada simulaciÃ³n*/
	double tiempoMax;
	/** Tiempo invertido en que el numBoidsOk lleguen al objetivo*/
	double tiempoInvertido;
	/** NÃºmero de boids que han alcanzado el objetivo*/
	int numBoidsOk;
	/** NÃºmero de boids que han alcanzado el objetivo y se consideran suficientes para detener
	 *  la simulaciÃ³n*/
	int numBoidsOkDeseados;
	/** NÃºmero de iteraciones que tarda la simulaciÃ³n en alcanzar la condiciÃ³n de Ã©xito*/
	int contIteraciones = 0;
	/** Distancia a la que se considera que se ha alcanzado el objetivo*/
	double distOk;
	/** NÃºmero de boids que forman la bandada*/
	int tamanoBandada;
	/** Vector donde se alamacenan los boids de la bandada*/
	Vector<Boid> bandada = new Vector<Boid>();
	/** Vector donde se almacenan los boids que han alcanzado el objetivo*/
	Vector<Boid> boidsOk = new Vector<Boid>();
	/** Vector con la informaciÃ³n de posiciÃ³n de los obstÃ¡culos del escenario*/
	Vector<Obstaculo> obstaculos = new Vector<Obstaculo>();
	/** Vector que contiene los puntos de diseÃ±o para la simulaciÃ³n por lotes*/
	Vector <Hashtable> vectorSim = new Vector<Hashtable>();
	Vector <TipoCamino> caminos = new Vector<TipoCamino>();
	private int incrNuevosBoids = 1;
	private int incrPensar = 1;
	private int contNuevosBoids = incrNuevosBoids ;
	private int contPensar = 0;
	private double comando = 0;
	public double anchoEscenario = 0;
	public double largoEscenario = 0;
	private LoggerArrayDoubles logPosturaCoche;
	private LoggerArrayDoubles logEstadistica;
	private double distOkAlOrigen = 5;
	/**
	 * radio en el que se busca el siguiente boid para formar el camino
	 */
	double umbralCercania = 8;//12;

	/**
	 * indica si existe una ruta con un tamaño mayor o igual a 2 puntos
	 */
	private boolean rutaParcial;
	/**
	 * 
	 */
	private boolean rutaCompleta;
	private double[] x = null;
	private double[] y = null;
	
	//Búsqueda A estrella
	
	Grid rejilla;
	
	//-------------Constructores---------------------------------------------------
	

	public Grid getRejilla() {
		return rejilla;
	}

	public void setRejilla(Grid rejilla) {
		this.rejilla = rejilla;
	}

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
		setDistOk(3);
		setNumBoidsOkDeseados(2);
		logPosturaCoche=LoggerFactory.nuevoLoggerArrayDoubles(this, "PosturaCoche");
		logPosturaCoche.setDescripcion("Coordenadas y yaw [x,y,yaw]");
		logEstadistica=LoggerFactory.nuevoLoggerArrayDoubles(this, "Estadistica");
		logEstadistica.setDescripcion(
				"Valores estadisticos del comportamiento del coche" +
				" [mediaVel,desvTipicaVel,mediaYaw,desvTipicaYaw,mediaAcel,desvTipicaAcel," +
				"mediaDistMin,desvTipicaDistMin]");
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
	
	//-------------MÃ©todos para el manejo de la bandada----------------------------
	
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
	
	public void crearBandada(int numBoids,double fechaNacimiento){
		if (getBandada().size() > 0)
			borrarBandada();
		setTamanoBandada(numBoids);
    	for (int i=0; i< tamanoBandada;i++){
    		getBandada().add(new Boid(new Matrix(2,1),new Matrix(2,1),new Matrix(2,1),
    				fechaNacimiento));
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
//				double vel[] = {Math.random(),Math.random()};
				double vel[] = {0,0};
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
			System.err.println("La bandada principal estÃ¡ vacÃ­a");
	}
	//-----------Fin de los mÃ©todos para manejar la bandada-----------------
	
	//-----------MÃ©todos para la simulaciÃ³n---------------------------------
	/**
	 * MÃ©todo que genera tantos obstÃ¡culos como numObst, con una magnitud de velocidad velMax
	 * con direcciÃ³n aleatoria. La posiciÃ³n de los obstÃ¡culos tambiÃ©n es aleatoria
	 */
	public void generaObstaculos(int numObst,double velMax){
		//Se eliminan los obstÃ¡culos que pudiera haber de anteriores simulaciones
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
			System.out.println("La ruta dinÃ¡mica estÃ¡ vacia");
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
//		System.out.println("acabÃ³ el control predictivo");
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
		if (!isRutaParcial()){
			velocidad = -1*paradaEmergencia;
		}else{
			velocidad = 3*paradaEmergencia;
		}
//		System.out.println("velocidad calculada "+velocidad);
		return velocidad;
	}
	
	public void moverPtoInicial(double tiempoActual,double Ts){
//		if(Math.abs(tiempoActual-tiempoAnt) > 500){
			double velocidad = calculaVelocidadCoche();
			if (rutaDinamicaSuave.size()<=3){
				comando  = calculaComandoVolante();
			}else{
//				tr = new Trayectoria(traduceRuta(rutaDinamica));
////			tr = new Trayectoria(tr,0.1);
//			tr.situaCoche(posInicial.get(0,0),posInicial.get(1,0));
//			contPred.setRuta(tr);
//				System.out.println("tamaño de la ruta suavizada " + rutaDinamicaSuave.size());
				double [][] trayecAux = traduceRuta(rutaDinamicaSuave);
				Trayectoria trayec = new Trayectoria(trayecAux);
				contPred.setCarroOriginal(this.modCoche);
				trayec.situaCoche(posInicial.get(0,0),posInicial.get(1,0));				
				contPred.setRuta(trayec);
//				contPred.setTs(Ts/1000);
				comando  = contPred.calculaComando();
				comando = sibtra.util.UtilCalculos.limita(comando,-Math.PI/6,Math.PI/6);
			}	
//			System.out.println("el comando calculado es " + comando);
			setTiempoAnt(tiempoActual);
//		}   
			modCoche.calculaEvolucion(comando,velocidad,Ts);
			posInicial.set(0,0,modCoche.getX());
			posInicial.set(1,0,modCoche.getY());
			logPosturaCoche.add(modCoche.getX(),modCoche.getY(),modCoche.getYaw());
//			System.out.println("yaw del coche "+modCoche.getYaw());
	}
	public double distObstaculoMasCercanoAlCoche(){
		double dist = Double.POSITIVE_INFINITY;
		double distMin = Double.POSITIVE_INFINITY;
		for (int i=0;i<obstaculos.size();i++){
			dist = posInicial.minus(obstaculos.elementAt(i).getPosicion()).norm2();
			if (dist < distMin){
				distMin = dist;
			}
		}
		return distMin;
	}
	public Matrix moverCocheSolitario(double Ts){
		//Seguir el objetivo
		Matrix velObj = new Matrix(2,1);
		Matrix velTotal = new Matrix(2,1);
		velObj = objetivo.minus(posCocheSolitario);
		velObj = velObj.times(Boid.pesoObjetivo);
		if (Math.abs(velObj.norm2()) > Boid.velMax)
			velObj = velObj.times(1/velObj.norm2()).times(Boid.velMax);
//		velObj = velObj.minus(velCocheSolitario);
		if (velObj.norm2() != 0)
			velObj = velObj.times(1/velObj.norm2()); // vector unitario
		/** Regla para esquivar los obstÃ¡culos*/
			double pos[] = {0,0};
			double zero[] = {0,0};
			Matrix cero = new Matrix(zero,2);
			Matrix c = new Matrix(pos,2);
			Matrix repulsion = new Matrix(zero,2);
			Matrix direcBoidObstaculo = new Matrix(zero,2);
			Matrix compensacion = new Matrix(zero,2);
			double dist = 0;
			double umbralEsquivar = Math.toRadians(20);
			double umbralCaso3 = -Math.toRadians(10);
			for (int i=0;i < obstaculos.size();i++){
				dist = obstaculos.elementAt(i).getPosicion().minus(posCocheSolitario).norm2();
				if (dist < Boid.radioObstaculo){
					repulsion = repulsion.minus(obstaculos.elementAt(i).getPosicion().minus(posCocheSolitario));
					//es el vector que apunta desde al boid hacia el obstÃ¡culo
					if (dist != 0){
						repulsion = repulsion.times(1/(dist)*(dist));
					}
					repulsion = repulsion.times(Boid.pesoObstaculo);
					//Dependiendo de la velocidad del obstÃ¡culo, de la posiciÃ³n del Boid
					//y de la posiciÃ³n del objetivo, se calcularÃ¡ una compensaciÃ³n lateral
					direcBoidObstaculo = repulsion.times(-1);
					double angVelObst = Math.atan2(obstaculos.elementAt(i).getVelocidad().get(1,0),
							obstaculos.elementAt(i).getVelocidad().get(0,0));
					double angDirecBoidObstaculo = Math.atan2(direcBoidObstaculo.get(1,0),
							direcBoidObstaculo.get(0,0));
					double angDirecObjetivo = Math.atan2(velObj.get(1,0),
							velObj.get(0, 0));
					double angCompensacion = 0;
					// Solo producen repulsiÃ³n aquellos obstÃ¡culos que se encuentren entre el objetivo
					// y el boid, los que quedan detrÃ¡s del boid no influencian
					if (UtilCalculos.diferenciaAngulos(angDirecObjetivo, angDirecBoidObstaculo)< 3*Math.PI/2){
						//Diferencia entre el Ã¡ngulo formado por el vector desde el boid hacia
						//el obstÃ¡culo y la velocidad del obstÃ¡culo y el Ã¡ngulo formado entre
						//el vector que va desde el boid hacia el objetivo y la velocidad del
						//obstÃ¡culo
						double angObsBoidObj = UtilCalculos.diferenciaAngulos(angVelObst,angDirecBoidObstaculo) -
								UtilCalculos.diferenciaAngulos(angVelObst, angDirecObjetivo);
						// caso en el que el boid y el obstÃ¡culo van a cruzar sus caminos 
						// en el futuro
//						if (UtilCalculos.diferenciaAngulos(angVelObst,angDirecBoidObstaculo) >=
//							UtilCalculos.diferenciaAngulos(angVelObst, angDirecObjetivo)){
						if (angObsBoidObj >= umbralCaso3){
//							if (UtilCalculos.diferenciaAngulos(angDirecBoidObstaculo, angDirecObjetivo) <= umbralEsquivar){
							if (angObsBoidObj > umbralEsquivar){// Por delante
//								System.out.println("va por delante del  obstÃ¡culo");
								compensacion.set(0,0,repulsion.get(1,0));
								compensacion.set(1,0,-repulsion.get(0,0));
								angCompensacion = Math.atan2(compensacion.get(1,0),
										compensacion.get(0,0));
								if (UtilCalculos.
										diferenciaAngulos(angVelObst,angCompensacion)>Math.toRadians(90)){
									//Si se da la condiciÃ³n lo cambiamos de sentido, si no se queda 
									//como se calculÃ³ antes del if
									compensacion.set(0,0,-repulsion.get(1,0));
									compensacion.set(1,0,repulsion.get(0,0));
								}
								
							}else{//Por detrÃ¡s
//								System.out.println("va por detrÃ¡s del  obstÃ¡culo");
								compensacion.set(0,0,repulsion.get(1,0));
								compensacion.set(1,0,-repulsion.get(0,0));
								angCompensacion = Math.atan2(compensacion.get(1,0),
										compensacion.get(0,0));
								if (UtilCalculos.
										diferenciaAngulos(angVelObst,angCompensacion)<Math.toRadians(90)){
									//Si se da la condiciÃ³n lo cambiamos de sentido, si no se queda 
									//como se calculÃ³ antes del if
									compensacion.set(0,0,-repulsion.get(1,0));
									compensacion.set(1,0,repulsion.get(0,0));
								}
//								sentidoCompensacionLateral = 1;
							}
							compensacion.timesEquals(Boid.pesoCompensacionLateral);
						
							c = c.plus(repulsion.plus(compensacion));
						}else{//Si no va a cruzarse con el obstÃ¡culo no se le aÃ±ade compensaciÃ³n lateral
							//ni repulsion						
							c = c.plus(cero);
						}
					}				
					
				}
			}
//			c = c.minus(velCocheSolitario);
			if (c.norm2() != 0)
				c = c.times(1/c.norm2()); // vector unitario
			velTotal = velObj.plus(c);
//			velTotal = velTotal.minus(velCocheSolitario);
//			velTotal = velObj;
			if (velTotal.norm2() != 0)
				velTotal = velTotal.times(1/velTotal.norm2()); // vector unitario
			double angVelCocheSolitario = Math.atan2(velCocheSolitario.get(1,0),
					velCocheSolitario.get(0,0));
//			System.out.println("Ã¡ngulo del coche "+ angVelCocheSolitario);
//			System.out.println("yaw del coche "+ cocheSolitario.getYaw());
			// descompongo el vector veltotal en sus componentes perpendiculares y paralelas
			// a la velocidad del coche. La paralela serÃ¡ la consigna de velocidad y la 
			// perpendicular serÃ¡ la consigna del volante
			double consVelocidadVec = velTotal.get(0,0)*Math.cos(-angVelCocheSolitario)-
			velTotal.get(1,0)*Math.sin(-angVelCocheSolitario);
//			double consVelocidadVec = velTotal.get(0,0);
			double consVelocidad = consVelocidadVec*3;// 3 es la velocidad mÃ¡xima
			double consVolanteVec = velTotal.get(0,0)*Math.sin(-angVelCocheSolitario)+
			velTotal.get(1,0)*Math.cos(-angVelCocheSolitario);
//			double consVolanteVec = velTotal.get(1,0);
			double consVolante = consVolanteVec*Math.toRadians(30);// a lo mejor hay que cambiar signo
			cocheSolitario.calculaEvolucion(consVolante,consVelocidad,Ts);
//			System.out.println("consigna volante: "+consVolante+"consigna velocidad: "+consVelocidad);
			posCocheSolitario.set(0,0,cocheSolitario.getX());
			posCocheSolitario.set(1,0,cocheSolitario.getY());
			velCocheSolitario.set(0,0,Math.cos(cocheSolitario.getYaw()));
			velCocheSolitario.set(1,0,Math.sin(cocheSolitario.getYaw()));
			return velTotal;
			
	}
	/**
	 * Calcula el desplazamiento y mueve cada uno de los Boids de la bandada. Se le pasa
	 * el Ã­ndice del lider de la iteraciÃ³n anterior
	 */
//	public int moverBoids(int indMinAnt){
	public void moverBoids(){
		int indLider = 0;
		double distMin = Double.POSITIVE_INFINITY;
		boolean liderEncontrado = false;
		contIteraciones++;
		//If para controlar la frecuencia a la que se aÃ±aden boids a la bandada
		if(contIteraciones > contNuevosBoids){
			for(int g=0;g<3;g++){				
//				double pos[] = {Math.abs(700*Math.random()),Math.abs(500*Math.random())};
//				double pos[] = {getBandada().lastElement().getPosicion().get(0,0)+10*Math.random(),
//						getBandada().lastElement().getPosicion().get(1,0)+10*Math.random()};
				double pos[] = {posInicial.get(0,0)+Math.random(), posInicial.get(1,0)+Math.random()};
				Matrix posi = new Matrix(pos,2);
				double vel[] = {0,0};
				Matrix velo = new Matrix(vel,2);
				double ace[] = {0,0};
				Matrix acel = new Matrix(ace,2);
				//Indicamos en que iteracion se crea el boid para despues calcular
				//su antiguedad
				getBandada().add(new Boid(posi,velo,acel,getContIteraciones()));
//				getBandada().add(new Boid(posi,velo,acel));
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
//			System.out.println("TamaÃ±o actual de la bandada " + getBandada().size());
		
			for (int j = 0;j<getBandada().size();j++){
				getBandada().elementAt(j).setConectado(false);
				getBandada().elementAt(j).setAntiguo((double)contIteraciones);
				getBandada().elementAt(j).calculaValoracion();				
				if(contIteraciones > contPensar){
					getBandada().elementAt(j).calculaMover(getBandada()
						,getObstaculos(),j,Boid.getObjetivo());					
				}
//				getBandada().elementAt(j).mover(getBandada()
//						,getObstaculos(),j,Boid.getObjetivo());
				getBandada().elementAt(j).mover();
				double dist = getBandada().elementAt(j).getDistObjetivo();
				// Deshabilitamos el liderazgo de la iteraciÃ³n anterior antes de retirar ningÃºn 
				// de la bandada por cercanÃ­a al objetivo				
//				Si estÃ¡ lo suficientemente cerca del objetivo lo quitamos de la bandada
				if (dist < distOk){
					getBandada().remove(j);//Simplemente lo quito, no guardo lo que hizo
//					getBandada().elementAt(j).setNumIteraciones(getContIteraciones());
//					traspasarBoid(j);
//					numBoidsOk++; // Incremento el numero de boids que han llegado al objetivo
				}
				// Buscamos al lider
//				if(j < getBandada().size()){
//					if (getBandada().elementAt(j).isCaminoLibre()){										
//						if (dist < distMin){
//							distMin = dist;
//							indLider = j;
//							liderEncontrado = true;
//						}
//					}
//				}
									
			}
			if (contIteraciones > contPensar){
				contPensar = contIteraciones + incrPensar;			
			}
			
//			if (indMinAnt<getBandada().size())
//				getBandada().elementAt(indMinAnt).setLider(false);
//			if (liderEncontrado && (indLider<getBandada().size())){
//				getBandada().elementAt(indLider).setLider(true);
//			}
		}
//		return indLider;				
	}
	

	/**
	 * MÃ©todo que utiliza la informaciÃ³n de un barrido de rangeFinder para posicionar 
	 * los obstÃ¡culos en el escenario
	 * @param ba Barrido de un rangeFinder, tanto LMS221 como LMS112
	 */
	public Vector<Obstaculo> posicionarObstaculos(BarridoAngular ba){
		getObstaculos().clear(); // Se eliminan los obstÃ¡culos que pudieran existir en el
								// escenario
		for (int i = 0; i<ba.numDatos();i++){
			double ang=ba.getAngulo(i);
			double dis=ba.getDistancia(i)*500;
//			System.out.println("distancia medida " + dis);
			double pos[] = {Math.abs(dis*Math.cos(ang)),Math.abs(dis*Math.sin(ang))};
			double vel[] = {0,0}; // En un futuro tal vez se tenga una estimaciÃ³n de la velocidad
			getObstaculos().add(new Obstaculo(new Matrix(pos,2),new Matrix(vel,2)));
//			getObstaculos().get(i).posicion.print(2,2);
			
		}
		return getObstaculos();
		//Usando un BarridoAngularIterator en lugar de BarridoAngular
//		while(ba.next()){
//			double ang=ba.angulo();
//			double dis=ba.distancia();
//			double pos[] = {dis*Math.cos(ang),dis*Math.sin(ang)};
//			double vel[] = {0,0}; // En un futuro tal vez se tenga una estimaciÃ³n de la velocidad
//			getObstaculos().add(new Obstaculo(new Matrix(pos,2),new Matrix(vel,2)));
//		}
	}
	
	/**
	 * Recorre el vector de obstÃ¡culos y dependiendo de los valores de velocidad y aceleraciÃ³n 
	 * de cada obstÃ¡culo se calcula el desplazamiento y se actualiza el valor de posiciÃ³n de 
	 * cada obstÃ¡culo
	 */
	public void moverObstaculos(){
		if(getObstaculos().size() != 0){
			for(int i = 0;i<getObstaculos().size();i++){	
				double gananciaVel = calculaParadaEmergencia(posInicial,
						getObstaculos().elementAt(i).getPosicion(),
						getObstaculos().elementAt(i).getVelocidad());
				getObstaculos().elementAt(i).mover(
						getObstaculos().elementAt(i).getRumboDeseado().times(gananciaVel),Ts);
				//-Control para que los obstÃ¡culos vuelvan a aparecer por el lado contrario-
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
	 * Dependiendo de la posiciÃ³n del vehÃ­culo el obstÃ¡culo se detendrÃ¡ o no
	 * @param Obs ObstÃ¡culo sobre el que se quiere calcular su velocidad 
	 * @return velocidad
	 */
	public double calculaParadaEmergencia(Matrix posObst,Matrix posObjMovil, Matrix velObjMovil){
		double veloObjMovil = 0;		
		// calculo el vector diferencia entre la pos del obstÃ¡culo
		// y la posicion inicial de salida de los boids, es decir, la posiciÃ³n del
		// vehÃ­culo
		Matrix difPosInicialObs = new Matrix(2,1);
		difPosInicialObs = posObst.minus(posObjMovil);
		double dist = difPosInicialObs.norm2();
		// calculo el Ã¡ngulo entre la velocidad del obstÃ¡culo y el vector diferencia 
		// anteriormente calculado para saber si el coche estÃ¡ dentro de la trayectoria
		// del obstÃ¡culo
		 // OJO CON EL MENOS DE LA COMPONENTE Y, EN ES SIST DE REF DE LA PANTALLA EL EJE Y
		 // ESTÃ� INVERTIDO!!
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
//			 Bucle while que realiza una simulaciÃ³n completa, es decir, hasta que lleguen
			// los boids especificados o hasta que se cumpla el tiempo mÃ¡ximo
		while ((tiempoInvertido < tiempoMax) && (numBoidsOk < numBoidsOkDeseados)){
//			indMinAnt =  moverBoids(indMinAnt);
			moverBoids();
			tiempoInvertido = (System.currentTimeMillis()-tiempoIni)/1000;
			contIteraciones++; // Llevamos la cuenta de las iteraciones del bucle principal de 
			// la simulaciÃ³n
//			bandada.add(new Boid(posInicial,new Matrix(2,1)));
//			getBandada().remove(ind);
//			System.out.println("tamaÃ±o de la bandada " + bandada.size());
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
	 * MÃ©todo que se encarga de crear la trayectoria hasta el objetivo usando la tÃ©cnica de
	 * la cadena de boids
	 * @param indLider Se le pasa el Ã­ndice del boid mÃ¡s cercano y con visiÃ³n directa hasta el objetivo
	 * @return
	 */
	
//	public Vector<Matrix> calculaRutaDinamica(int indLider){	
	public Vector<Matrix> calculaRutaDinamica(){
		rutaDinamica.clear();
		setRutaParcial(false);
		setRutaCompleta(false);
//		int boidActual = indLider;
		int boidActual = 0;
		int boidAux = 0;
		int cont = 0;
		boolean encontrado = false;
		double valoracion = Double.NEGATIVE_INFINITY;		
		double radioCentroMasas = umbralCercania*0.2;//*0.5;
//		System.out.println("EmpezÃ³ nueva ruta");
		rutaDinamica.add(posInicial);
		Matrix puntoActual = new Matrix(2,1);
		puntoActual.set(0,0,posInicial.get(0,0));
		puntoActual.set(1,0,posInicial.get(1,0));
		while (cont < getBandada().size()){
			cont++;
			encontrado=false;
			for (int i=0;i < bandada.size();i++){				
				if (i != boidActual){// No se comprueba consigo mismo
					if(!getBandada().elementAt(i).isConectado()){// El boid elegido no puede estar
						// conectado con otro
//						double dist = bandada.elementAt(boidActual).getPosicion().minus(getBandada().elementAt(i).getPosicion()).norm2();
						double dist = puntoActual.minus(getBandada().elementAt(i).getPosicion()).norm2();
						if (dist < umbralCercania){// Tiene que estar lo suficientemente cerca
							boolean caminoOcupado = false;
							// Calculamos la recta entre ambos boids
//							Line2D recta = 
//								new Line2D.Double(getBandada().elementAt(boidActual).getPosicion().get(0,0),
//										getBandada().elementAt(boidActual).getPosicion().get(1,0),
//										getBandada().elementAt(i).getPosicion().get(0,0),
//										getBandada().elementAt(i).getPosicion().get(1,0));
							Line2D recta = 
								new Line2D.Double(puntoActual.get(0,0),puntoActual.get(1,0),
										getBandada().elementAt(i).getPosicion().get(0,0),
										getBandada().elementAt(i).getPosicion().get(1,0));
							for (int j=0;j < obstaculos.size();j++){
								double distObs = obstaculos.elementAt(j).getPosicion().minus(
										getBandada().elementAt(boidActual).getPosicion()).norm2();
								if (distObs < umbralCercania){
									if (!caminoOcupado)// SÃ³lo se calcula la intersecciÃ³n mientras el camino siga sin ocupar
										caminoOcupado = recta.intersects(obstaculos.elementAt(j).getForma());
								}							
							}
							if(!caminoOcupado){
								if(getBandada().elementAt(i).getValoracion() > valoracion){
									boidAux = i;
									valoracion = getBandada().elementAt(i).getValoracion();
									encontrado = true;
								}
//								System.out.println("encontrÃ³ compaÃ±ero");
							}
						}
					}
				}
			}
			if(encontrado){
				getBandada().elementAt(boidAux).setConectado(true);
				if (getBandada().elementAt(boidAux).getPosicion().minus(objetivo).norm2()<distOkAlOrigen){
					setRutaCompleta(true);
					rutaDinamica.add(objetivo);
				}else{
//					getBandada().elementAt(boidAux).setExperiencia(1);
//					System.out.println("La valoracion es : " + valoracion);
					rutaDinamica.add(getBandada().elementAt(boidAux).getPosicion());
//					rutaDinamica.add(getBandada().elementAt(boidAux).calculaCentroMasas(getBandada(),radioCentroMasas));
					boidActual = boidAux;
					puntoActual.set(0,0,getBandada().elementAt(boidActual).getPosicion().get(0,0));
					puntoActual.set(1,0,getBandada().elementAt(boidActual).getPosicion().get(1,0));
//					System.out.println("saltÃ³ al siguiente boid");
				}

			}
			
		}
//		rutaDinamica = mejoraRuta(rutaDinamica);
//		boolean caminoOcupadoPosIni = false;
//		// Calculamos la recta entre el ultimo punto de la ruta dinamica y la posiciÃ³n final
//		Line2D recta = 
//			new Line2D.Double(rutaDinamica.lastElement().get(0,0),
//					rutaDinamica.lastElement().get(1,0),
//					objetivo.get(0,0),objetivo.get(1,0));
//		for (int j=0;j < obstaculos.size();j++){// Se comprueba con todos los obstÃ¡culos
////			double distObs = obstaculos.elementAt(j).getPosicion().minus(
////					rutaDinamica.lastElement()).norm2();
////			if (distObs < umbralCercania){
//				if (!caminoOcupadoPosIni)// SÃ³lo se calcula la intersecciÃ³n mientras el camino siga sin ocupar
//					caminoOcupadoPosIni = recta.intersects(obstaculos.elementAt(j).getForma());
////			}							
//		}
//		if(!caminoOcupadoPosIni){
//			rutaDinamica.add(objetivo);
//			setRutaCompleta(true);
//		}		
//		System.out.println("acabÃ³ la ruta");
		if (rutaDinamica.size()>=2){
			setRutaParcial(true);
//			tr = new Trayectoria(traduceRuta(rutaDinamica));
////			tr = new Trayectoria(tr,0.1);
//			tr.situaCoche(posInicial.get(0,0),posInicial.get(1,0));
//			contPred.setRuta(tr);
		}
//		rutaDinamica = mejoraRuta(rutaDinamica);
		return rutaDinamica;
	}
	
	public Vector<Matrix> busquedaAEstrella(){
		
		boolean tentative_is_better = false;
		boolean caminoCompleto = false;
		double minF_score = Double.POSITIVE_INFINITY;
		int indMin = 0;
		Vector<Matrix> camino = new Vector<Matrix>();
		Vector<Boid> openSet = new Vector<Boid>();
		Vector<Boid> closedSet = new Vector<Boid>();		
		openSet.clear();
		closedSet.clear();
		// reseteamos los valores de las variables de binarias de la bandada
		for (int h=0;h<bandada.size();h++){
			bandada.elementAt(h).setOpenSet(false);
			bandada.elementAt(h).setClosedSet(false);
		}
		//Añadimos un boid en la posición del coche para que sea el nodo inicial 
		double pos[] = {posInicial.get(0,0), posInicial.get(1,0)};
		Matrix posi = new Matrix(pos,2);
		double vel[] = {0,0};
		Matrix velo = new Matrix(vel,2);
		double ace[] = {0,0};
		Matrix acel = new Matrix(ace,2);
		//Calculamos su h_score y su f_score y se lo asignamos, su g_score es cero al ser el nodo inicial
		Boid nodo_inicial = new Boid(posi,velo,acel,getContIteraciones());
		Boid actual = nodo_inicial;
		nodo_inicial.setG_score(0);
		nodo_inicial.setH_score(nodo_inicial.getDistObjetivo());//Distancia euclídea hasta el objetivo
		nodo_inicial.calculaF_score();
		//y lo añadimos al openSet y a la bandada
		this.getBandada().add(nodo_inicial);
		nodo_inicial.setOpenSet(true);
		nodo_inicial.setClosedSet(false);
		openSet.add(nodo_inicial);
		
		//Bucle principal del algoritmo
		while (!openSet.isEmpty()){//Mientras queden nodos en el openset el algoritmo continua
			//el nodo actual será aquel que tenga el f_score más bajo de entre los que se encuentran en el openSet
			minF_score = Double.POSITIVE_INFINITY;
//			System.out.println("tamaño del openSet = "+openSet.size());
			for (int i=0;i<openSet.size();i++){
				if(openSet.elementAt(i).getF_score() < minF_score) {
					indMin=i;
					minF_score=openSet.elementAt(i).getF_score();
				}								
			}
			actual = openSet.elementAt(indMin);
//			System.out.println("posición del actual "+actual.getPosicion().get(1,0)+" "+actual.getPosicion().get(1,0));
			// no estoy seguro de que podamos asignar esto así...
			// Si el boid acutal está lo suficientemente cerca del objetivo damos por concluida la búsqueda
			if (actual.getDistObjetivo() < distOkAlOrigen){
				// reconstruir el camino,acaba el algoritmo
//				rutaDinamica.clear();
//				rutaDinamica = reconstruirCaminoAEstrella(actual);
				camino = reconstruirCaminoAEstrella(actual);
				caminoCompleto = true;
				System.out.println("se encontró un camino completo");
			}
			//Quitamos el nodo actual del openSet 
			openSet.remove(indMin); //También existe un método para quitar un elemento especificando que objeto hay que quitar
			//y lo añadimos al closedSet
			actual.setOpenSet(false);
			actual.setClosedSet(true);
			closedSet.add(actual);
			for (int j = 0;j<getBandada().size();j++){				
				//Los nodos que  se estudian tienen que ser vecinos del actual
				if (actual.getPosicion().minus(getBandada().elementAt(j).getPosicion()).norm2() <= umbralCercania){
					//g_score_tentativo es la suma del g_score del actual más la distancia entre al actual
					//y el vecino en cuestión
					if (getBandada().elementAt(j).isClosedSet()){
//						System.out.println("está en el closedSet");
						continue; // si el nodo está en el closedSet no hacemos nada con el y seguimos mirando
						
					}
					double g_score_tentativo = actual.getG_score() +
							actual.getPosicion().minus(getBandada().elementAt(j).getPosicion()).norm2();
					if (!getBandada().elementAt(j).isOpenSet()){//Comprobamos si el vecino está en el openSet
//						System.out.println("el vecino no está en el openset");
						//si no está lo metemos en el openSet
						getBandada().elementAt(j).setOpenSet(true);
						getBandada().elementAt(j).setClosedSet(false);
						openSet.add(getBandada().elementAt(j));
						//Calculamos su h_score (distancia euclídea hasta el objetivo) y se lo asignamos
						getBandada().elementAt(j).setH_score(getBandada().elementAt(j).getDistObjetivo());
						tentative_is_better = true;
					}else if (g_score_tentativo < getBandada().elementAt(j).getG_score()){
//						System.out.println("el vecino está en el openset y la tentativa es mejor");
						//la tentativa es mejor si el g_score tentativo es mejor que el g_score del vecino
						tentative_is_better = true;
					}else{// si no es así la tentativa es peor
//						System.out.println("el vecino está en el openset y la tentativa es peor");
						tentative_is_better = false;
					}
					if (tentative_is_better){
//						System.out.println("la tentativa es mejor");
						//Indicamos desde que nodo (boid) hemos llegado a este vecino
						getBandada().elementAt(j).setCame_from(actual);
						getBandada().elementAt(j).setG_score(g_score_tentativo);
						getBandada().elementAt(j).calculaF_score();
						
					}
//					System.out.println("diferencia de f_score entre bandada y openSet "+(getBandada().elementAt(j).getF_score()-openSet.lastElement().getF_score()));
				}
			}
		}
		if (!caminoCompleto){
//			rutaDinamica.clear();
//			rutaDinamica = reconstruirCaminoAEstrella(actual);
			camino = reconstruirCaminoAEstrella(actual);
			System.out.println("no se logró un camino completo");
		}
		return camino;
	}
	
	public Vector<Matrix> reconstruirCaminoAEstrella(Boid actual){
		Vector<Matrix> camino = new Vector<Matrix>();
		Boid aux = actual;
		while(aux.getCame_from() != null){
			camino.add(aux.getPosicion());
			aux = aux.getCame_from();			
		}
		System.out.println("tamaño del camino a estrella "+ camino.size());
		return camino;
	}
	
	/**
	 * 
	 * @param ruta vector de Matriz con los puntos de la ruta dinámica
	 * @return Devuelve true si la ruta no es válida, es decir, un obstáculo está intersectándola
	 */
	public boolean compruebaRuta(){
		boolean hayObs = false;
		if (rutaDinamica.size()<1){// si no hay ruta no se comprueba
			return hayObs = true; // Para que si no hay ruta la calcule
		}else{
			//sigue mientras haya puntos en la ruta y no se haya encontrado un obstáculo
			for (int i=0;(i<rutaDinamica.size()-1) && (!hayObs);i++){ 
				for (int j=0;j < obstaculos.size();j++){
					double distObs = obstaculos.elementAt(j).getPosicion().minus(
							rutaDinamica.elementAt(i)).norm2(); //Calculamos la distancia hasta los obstáculos
					double umbralCercania = rutaDinamica.elementAt(i).minus(
							rutaDinamica.elementAt(i+1)).norm2();//Calculamos la distancia entre dos puntos consecutivos de la ruta
					//Hallamos la recta entre dos puntos consecutivos para estudiar si se intersecta con algún obstáculo
					Line2D recta = 
							new Line2D.Double(rutaDinamica.elementAt(i).get(0,0),rutaDinamica.elementAt(i).get(1,0),
									rutaDinamica.elementAt(i+1).get(0,0),rutaDinamica.elementAt(i+1).get(1,0));
					if (distObs <= umbralCercania){					
							hayObs = recta.intersects(obstaculos.elementAt(j).getForma());
							if (hayObs)
								break;
					}							
				}
			}
		}
		
		return hayObs;
	}
	/**
	 * MÃ©todo que simplifica la ruta calculada por calculaRutaDinamica
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
	 * MÃ©todo que traduce la informaciÃ³n de la ruta (vector de matrices) a un array de varias dimensiones
	 * @param ruta
	 * @return
	 */
	public double[][] traduceRuta(Vector<Matrix> ruta){
		if (ruta.size()==0){
			throw new IllegalArgumentException("la ruta tiene que tener puntos para poderla traducir");
		}
		double [][] arrayRuta = new double[ruta.size()][2];
		for (int i=0;i<=ruta.size()-1;i++){
			arrayRuta[i][0] = ruta.elementAt(i).get(0,0);
			arrayRuta[i][1] = ruta.elementAt(i).get(1,0);
//			System.out.println("en traduceRuta x e y son "+arrayRuta[i][0]+" "+arrayRuta[i][0]);
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
	         switch (indice){//Dependiendo de la etiqueta se varÃ­a un parÃ¡metro u otro
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
	
	
	public Vector<Matrix> suavizador(double distPuntos){
		double[] x = new double[rutaDinamica.size()];
		double[] y = new double[rutaDinamica.size()];		
		for(int i=0;i<rutaDinamica.size();i++) {
			x[i]=rutaDinamica.elementAt(i).get(0,0);
			y[i]=rutaDinamica.elementAt(i).get(1,0);	
		}		
		double xMin = UtilCalculos.minimo(Double.POSITIVE_INFINITY,x);
		double xMax = UtilCalculos.maximo(Double.NEGATIVE_INFINITY,x);
		int numPuntos = (int) Math.round((xMax-xMin)/distPuntos);
//		double[][] rutaSuavizada = new double[numPuntos][2];
		CubicSpline spline = new CubicSpline(x,y);
		rutaDinamicaSuave.clear();
				
		for (int i=0;i<numPuntos-1;i++){
//			rutaSuavizada[i][1] = xMin+i*distPuntos;
//			rutaSuavizada[i][2] = spline.interpolate(xMin+i*distPuntos);
			Matrix puntoActual = new Matrix(2,1);
			puntoActual.set(0,0,xMin+i*distPuntos);
			puntoActual.set(1,0,spline.interpolate(xMin+i*distPuntos));
			rutaDinamicaSuave.add(puntoActual);
		}		
		return rutaDinamicaSuave;
	}
	
	//-----------------------------------------------------------------------
	//-------------------Getters y Setters-----------------------------------
	//-----------------------------------------------------------------------
	//
	
	public double getUmbralCercania() {
		return umbralCercania;
	}

	public void setUmbralCercania(double umbralCercania) {
		this.umbralCercania = umbralCercania;
	}
	
	public Vector<Matrix> getRutaDinamicaSuave() {
		return rutaDinamicaSuave;
	}

	public void setRutaDinamicaSuave(Vector<Matrix> rutaDinamicaSuave) {
		this.rutaDinamicaSuave = rutaDinamicaSuave;
	}
	
	public Vector<Matrix> getRutaDinamica() {
		return rutaDinamica;
	}

	public void setRutaDinamica(Vector<Matrix> rutaDinamica) {
		this.rutaDinamica = rutaDinamica;
	}

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
	
	public void setContIteraciones(int contIteraciones) {
		this.contIteraciones = contIteraciones;
		this.contNuevosBoids = contIteraciones;
		this.contPensar = contIteraciones;
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
		this.cocheSolitario.setPostura(posInicial.get(0,0),posInicial.get(1,0),0);
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
	
	public LoggerArrayDoubles getLogEstadistica() {
		return logEstadistica;
	}
	
	public double getTiempoInvertido() {
		return tiempoInvertido;
	}
	
	public double getAnchoEscenario() {
		return anchoEscenario;
	}
	
	public Coche getCocheSolitario() {
		return cocheSolitario;
	}

	public void setCocheSolitario(Coche cocheSolitario) {
		this.cocheSolitario = cocheSolitario;
	}
	
	public Matrix getPosCocheSolitario() {
		return posCocheSolitario;
	}

	public void setPosCocheSolitario(Matrix posCocheSolitario) {
		this.posCocheSolitario = posCocheSolitario;
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
	
	public void creaRejilla(){
		rejilla = new Grid(0.5, getLargoEscenario(), getAnchoEscenario());
		for (int i=0; i < getObstaculos().size();i++){
			double posXObs = getObstaculos().elementAt(i).getPosicion().get(0,0);
			double posYObs = getObstaculos().elementAt(i).getPosicion().get(1,0);
			double dimensionX = getObstaculos().elementAt(i).getLado();
			double dimensionY = getObstaculos().elementAt(i).getLado();
//			System.out.println("creamos la rejilla ");
			rejilla.addObstacle(posXObs, posYObs, dimensionX, dimensionY);
		}
		
	}

	public double getTs() {
		return Ts;
	}

	public void setTs(double ts) {
		Ts = ts;
	}
	
	public boolean isRutaParcial() {
		return rutaParcial;
	}

	public void setRutaParcial(boolean rutaParcial) {
		this.rutaParcial = rutaParcial;
	}
	
	public boolean isRutaCompleta() {
		return rutaCompleta;
	}

	public void setRutaCompleta(boolean rutaCompleta) {
		this.rutaCompleta = rutaCompleta;
	}
	
	public ControlPredictivo getContPred() {
		return contPred;
	}
	
	//----------------------------------------------------------------------------
	//----------------Final de los Getters y Setters------------------------------
	//----------------------------------------------------------------------------
}
