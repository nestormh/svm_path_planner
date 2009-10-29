package boids;

import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.awt.geom.QuadCurve2D;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

import javax.swing.JFileChooser;

import Jama.Matrix;

public class Simulador {
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
	/** Número de boids que han alcanzado el objetivo yse consideran suficientes para detener
	 *  la simulación*/
	int numBoidsOkDeseados;
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
	
	//-------------Constructores---------------------------------------------------
	
	/**Constructor por defecto*/
	public Simulador(){
		setTamanoBandada(20);
		crearBandada(getTamanoBandada());
		posicionarBandada(new Matrix(2,1));
		setObjetivo(new Matrix(2,1));
		setTiempoMax(5);
		setDistOk(50);
		setNumBoidsOkDeseados(2);
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
    		getBandada().add(new Boid(new Matrix(2,1),new Matrix(2,1)));	
    	}
	}
	public void crearBandada(){
//		if (getBandada().size() > 0)
//			borrarBandada();
		getBandada().clear();
    	for (int i=0; i< tamanoBandada;i++){
    		getBandada().add(new Boid(new Matrix(2,1),new Matrix(2,1)));	
    	}
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
		setPosInicial(puntoIni);
		if (getBandada().size()>0){
			for (int i=0;i<getBandada().size();i++){				
//				double pos[] = {e.getX()+Math.random()*getTamanoBan()*2, e.getY()+Math.random()*getTamanoBan()*2};
				double pos[] = {posInicial.get(0,0)+Math.random(), posInicial.get(1,0)+Math.random()};
				Matrix posi = new Matrix(pos,2);
				double vel[] = {Math.random(),Math.random()};
				Matrix velo = new Matrix(vel,2);
				this.getBandada().elementAt(i).resetRuta();
				this.getBandada().elementAt(i).getForma().transform(AffineTransform.getTranslateInstance(pos[0]-getBandada().elementAt(i).getPosicion().get(0,0),
						pos[1]-getBandada().elementAt(i).getPosicion().get(1,0)));
				this.getBandada().elementAt(i).setPosicion(posi);			
				this.getBandada().elementAt(i).setVelocidad(velo);
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
	 * Calcula el desplazamiento y mueve cada uno de los Boids de la bandada. Se le pasa
	 * el índice del lider de la iteración anterior
	 */
	public int moverBoids(int indMinAnt){
		int indMin = 0;
		double distMin = Double.POSITIVE_INFINITY;
		boolean liderEncontrado = false;
		// Iteramos sobre toda la bandada
		if (getBandada().size() != 0){
			for (int j = 0;j<getBandada().size();j++){
				getBandada().elementAt(j).mover(getBandada()
						,getObstaculos(),j,Boid.getObjetivo());
				// Buscamos al lider
				if (getBandada().elementAt(j).isCaminoLibre()){
					double dist = getBandada().elementAt(j).getDistObjetivo();
					if (dist < distMin){
						distMin = dist;
						indMin = j;
						liderEncontrado = true;
					}
//					Si está lo suficientemente cerca del objetivo lo quitamos de la bandada
					if (dist < distOk){
						traspasarBoid(j);
						numBoidsOk++; // Incremento el numero de boids que han llegado al objetivo
					}
				}					
			}
			if (indMinAnt<getBandada().size())
				getBandada().elementAt(indMinAnt).setLider(false);
			if (liderEncontrado && (indMin<getBandada().size())){
				getBandada().elementAt(indMin).setLider(true);
			}
		}
		return indMin;				
	}

	public void simuPorLotes(){
		int indMinAnt = 0;
		double tiempoIni = System.currentTimeMillis();
		tiempoInvertido = 0;
		setNumBoidsOk(0);
//			 Bucle while que realiza una simulación completa, es decir, hasta que lleguen
			// los boids especificados o hasta que se cumpla el tiempo máximo
		while ((tiempoInvertido < tiempoMax) && (numBoidsOk < numBoidsOkDeseados)){
			indMinAnt =  moverBoids(indMinAnt);
			tiempoInvertido = (System.currentTimeMillis()-tiempoIni)/1000;
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
		System.out.println("Ya acabó el simuporlotes");
		System.out.println(getNumBoidsOk());
	}
	
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
	
	public void configurador(Hashtable designPoint,String[] nomParam){
		for (Enumeration e = designPoint.keys() ; e.hasMoreElements() ;) {
//	         System.out.println(e.nextElement());
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
	         		  crearBandada();
	         		  break;
	         }
	     }
	}
	
	//-----------------------------------------------------------------------
	//-------------------Getters y Setters-----------------------------------
	//-----------------------------------------------------------------------
	
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

	public Matrix getObjetivo() {
		return objetivo;
	}

	public void setObjetivo(Matrix objetivo) {
		this.objetivo = objetivo;
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
	
	//----------------------------------------------------------------------------
	//----------------Final de los Getters y Setters------------------------------
	//----------------------------------------------------------------------------
}
