package boids;

import java.awt.Color;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.io.Serializable;
import java.util.Vector;

import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;

import com.bruceeckel.swing.Console;

import Jama.Matrix;

public class Boid implements Serializable{
	/**Vector con las componentes de velocidad del boid*/
	Matrix velocidad;
	/**Vector con las componentes de posicion del boid*/
	Matrix posicion;
	/**Objeto gráfico que representará al boid*/
	GeneralPath triangulo;
	/**Linea que muestra la dirección del desplazamiento del boid*/
	Line2D lineaDireccion = new Line2D.Double();
	Vector<Matrix> rutaBoid = new Vector<Matrix>();
	public boolean lider = false;
	static double pesoCohesion = 0.05;
	static double pesoSeparacion = 10;
	static double pesoAlineacion = 2;
	static double pesoObjetivo = 1;
	static double pesoObstaculo = 10;
	static double pesoLider = 1;
	static double velMax = 15;
	static double coorObjetivo[] = {800,800};
	static Matrix objetivo = new Matrix(coorObjetivo,2);
	/**Constructor donde se inicializa la posición y velocidad de cada boid,
	 * además de el objeto gráfico que lo representará*/
	public Boid(Matrix posicion, Matrix velocidad) {
		this.velocidad = velocidad;
		this.posicion = posicion;
		lineaDireccion.setLine(this.posicion.get(0,0),this.posicion.get(1,0),
				this.velocidad.get(0,0),this.velocidad.get(1, 0));
		/**Inicialización del aspecto gráfico del cuerpo del boid*/
		float ptosX[] = {0,-2,2};
		float ptosY[] = {1,-1,-1};
		triangulo = new GeneralPath(GeneralPath.WIND_NON_ZERO,ptosX.length);
		triangulo.moveTo (ptosX[0], ptosY[0]);

		for (int index = 1; index < ptosX.length; index++) {
		 	 triangulo.lineTo(ptosX[index], ptosY[index]);
		};
		triangulo.closePath();
		triangulo.transform(AffineTransform.getTranslateInstance(posicion.get(0,0),posicion.get(1,0)));
		this.nuevoPuntoRuta(this.getPosicion());
	}
	
	/**Método optimizado para calcular la cohesión, alineación y separación para un boid. 
	 * Sólo usa un bucle for, y no una para cada regla*/
	
	public Matrix aliCoheSep(Vector<Boid> bandada,int indBoid){
		double pos[] = {0,0};
		Matrix velResultante = new Matrix(2,1);
		Matrix velMedia = new Matrix(2,1);
		Matrix centroMasa = new Matrix(2,1);
		Matrix velCohesion = new Matrix(pos,2);
		Matrix separa = new Matrix(pos,2);
		boolean liderCerca = false;
		int indLider = 0;
		int cont = 0;
		// Bucle que recorre toda la bandada
		for (int i=0;i < bandada.size();i++){
			if (i != indBoid){
				if (Math.abs(bandada.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()) < 30)
					velMedia = velMedia.plus(bandada.elementAt(i).getVelocidad());
				// if para la alineacion
				if (!this.isLider()){
					if (Math.abs(bandada.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()) < 100){
						if (bandada.elementAt(i).isLider()){
							liderCerca = true;
							indLider = i;
							break;
						}
						centroMasa = centroMasa.plus(bandada.elementAt(i).getPosicion());
						cont++;
					}
				} // if (!this.isLider()) para la cohesion
				if (Math.abs(bandada.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()) < 30){
					separa = separa.minus(bandada.elementAt(i).getPosicion().minus(this.getPosicion()));
					separa = separa.times(pesoSeparacion);
				}// if para la separacion
			} // if (i != indBoid) para todas las reglas
		} // for principal
		
		// calculos para la velocidad de alineación
		velMedia = velMedia.timesEquals(1/bandada.size()-1);
		velMedia = velMedia.minus(this.getVelocidad());
		velMedia = velMedia.times(pesoAlineacion);
		// calculos para la cohesión
		if (cont != 0 && liderCerca == false){
			centroMasa = centroMasa.timesEquals(1/cont);
			velCohesion = (centroMasa.minus(this.getPosicion())).times(pesoCohesion);
		}
		if(liderCerca == true){
			velCohesion = (bandada.elementAt(indLider).getPosicion().minus(this.getPosicion())).times(pesoLider);
		}
		velResultante = (velMedia.plus(velCohesion)).plus(separa);
		return velResultante;
	}
	/** Esta regla genera un vector velocidad que hace que el boid se agrupe
	 *  con sus compañeros de bandada*/
	public Matrix cohesion(Vector<Boid> bandada,int indBoid){
		double pos[] = {0,0};
		boolean liderCerca = false;
		int indLider = 0;
		Matrix velCohesion = new Matrix(pos,2);
		int cont = 0;
		Matrix centroMasa = new Matrix(pos,2);		
		if (!this.isLider()){ // El lider no se cohesiona con nadie
			for (int i=0;i < bandada.size();i++){
				if (i != indBoid)
					if (Math.abs(bandada.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()) < 100){
						if (bandada.elementAt(i).isLider()){
							liderCerca = true;
							indLider = i;
							break;
						}
						centroMasa = centroMasa.plus(bandada.elementAt(i).getPosicion());
						cont++;
					}
			}
			if (cont != 0 && liderCerca == false){
				centroMasa = centroMasa.timesEquals(1/cont);
				velCohesion = (centroMasa.minus(this.getPosicion())).times(pesoCohesion);
			}
			if(liderCerca == true){
				velCohesion = (bandada.elementAt(indLider).getPosicion().minus(this.getPosicion())).times(pesoLider);
			}			
		}
		return velCohesion;
	}
	
	/**Los boids intentan mantener la velocidad media de la bandada*/
	
	public Matrix alineacion(Vector<Boid> bandada,int indBoid){
		double pos[] = {0,0};
		Matrix velMedia = new Matrix(pos,2);
		for (int i=0;i < bandada.size();i++){
			if (i != indBoid)
				if (Math.abs(bandada.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()) < 50)
					velMedia = velMedia.plus(bandada.elementAt(i).getVelocidad());
		}
		velMedia = velMedia.timesEquals(1/bandada.size()-1);
		velMedia = velMedia.minus(this.getVelocidad());
		velMedia = velMedia.times(pesoAlineacion);
		return velMedia;
	}
	
	/**Los boids intentan no chocarse entre si*/
	
	public Matrix separacion(Vector<Boid> bandada,int indBoid){
		double pos[] = {0,0};
		Matrix separa = new Matrix(pos,2);
		for (int i=0;i < bandada.size();i++){
			if (i!=indBoid)
				if (Math.abs(bandada.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()) < 30){
					separa = separa.minus(bandada.elementAt(i).getPosicion().minus(this.getPosicion()));
					separa = separa.times(pesoSeparacion);
				}
		}				
		return separa;
	}
	
	/**Regla que permite fijar un objetivo para que los boids lo persigan*/
	
	public Matrix seguirObjetivo(Vector<Boid> bandada,int indBoid,Matrix obj){
		Matrix velObj = new Matrix(2,1);
		velObj = obj.minus(bandada.elementAt(indBoid).getPosicion());
		if(this.isLider()) // El lider es atraido con más fuerza por el objetivo
			velObj = velObj.times(pesoObjetivo*10);
		else
			velObj = velObj.times(pesoObjetivo);
		return velObj;
	}
	
	/** Regla para no permitir grandes velocidades*/ 
	
	public Matrix limitaVelocidad(Matrix vel){
		Matrix velLimitada = new Matrix(2,1);
		vel.print(10,3);
		velLimitada = vel;
		if (Math.abs(vel.norm2()) > velMax)
			velLimitada = vel.times(1/vel.norm2()).times(velMax);
		return velLimitada;
	}
	
	/** Regla para esquivar los obstáculos*/
	
	public Matrix evitaObstaculo(Vector<Obstaculo> obstaculos,Boid b){
		double pos[] = {0,0};
		Matrix c = new Matrix(pos,2);
		for (int i=0;i < obstaculos.size();i++){
			if (Math.abs(obstaculos.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()) < 40){
				c = c.minus(obstaculos.elementAt(i).getPosicion().minus(this.getPosicion()));
				c = c.times(pesoObstaculo);
//				c = c.times(1/Math.abs(obstaculos.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()));
			}
		}				
		return c;
	}
	
	/** Método que calcula todas las reglas para cada Boid, las suma vectorialmente
	 * 	, calcula el desplazamiento y lo realiza*/
	
	public void mover(Vector<Boid> bandada,Vector<Obstaculo> obstaculos,int indBoid, Matrix obj){
		Matrix desp = new Matrix(2,1);
//		Matrix despCohesion = new Matrix(2,1);
//		Matrix despSeparacion = new Matrix(2,1);
//		Matrix despAlineacion = new Matrix(2,1);
		Matrix despObjetivo = new Matrix(2,1);
		Matrix despAliCoheSep = new Matrix(2,1);
		Matrix despObstaculo = new Matrix(2,1);
//		despCohesion = cohesion(bandada, indBoid);
//		despSeparacion = separacion(bandada, indBoid);
//		despAlineacion = alineacion(bandada, indBoid);
		despAliCoheSep = aliCoheSep(bandada, indBoid);
		despObjetivo = seguirObjetivo(bandada,indBoid,obj);
		despObstaculo = evitaObstaculo(obstaculos,bandada.elementAt(indBoid));
		desp = limitaVelocidad(((despAliCoheSep.plus(despObjetivo)).plus(despObstaculo)).plus(this.getVelocidad()));
//		desp = limitaVelocidad(despCohesion.plus(despSeparacion).plus(despAlineacion).plus(despObjetivo).plus(despObstaculo).plus(this.getVelocidad()));
		this.getForma().transform(AffineTransform.getTranslateInstance(desp.get(0,0), desp.get(1,0)));
		this.setVelocidad(desp);
		this.setPosicion(this.getPosicion().plus(this.getVelocidad()));
		setLineaDireccion(getPosicion().get(0,0),getPosicion().get(1,0),
				(getPosicion().plus(getVelocidad().times(2))).get(0,0),
				(getPosicion().plus(getVelocidad().times(2))).get(1,0));
	}
	
	private void nuevoPuntoRuta(Matrix pto){
		this.rutaBoid.add(pto);
	}
	
	public Vector<Matrix> getRutaBoid(){
		return rutaBoid;
	}
	 public void resetRuta(){
		 rutaBoid.clear();
	 }
	
	public Matrix getPosicion() {
		return posicion;
	}
	
	public void setPosicion(Matrix pos) {
		this.posicion = pos;
		nuevoPuntoRuta(this.posicion);
	}
//	public void setPosicion(double x, double y) {
//		this.posicion.set(0,0,x);
//		this.posicion.set(1,0,y);
//	}
	
	public Matrix getVelocidad() {
		return velocidad;
	}
	
	public void setVelocidad(Matrix vel) {
		this.velocidad = vel;
	}
//	public void setVelocidad(double velX, double velY) {
//		this.velocidad.set(0,0,velX);
//		this.velocidad.set(1,0,velY);
//	}
	
	public GeneralPath getForma(){
		return triangulo;
	}
	
	public Line2D getLineaDireccion() {
		return lineaDireccion;
	}
	public void setLineaDireccion(Line2D direccion) {
		this.lineaDireccion = direccion;
	}
	public void setLineaDireccion(double xPtoa,double yPtoa,double xPtob, double yPtob) {
		this.lineaDireccion.setLine(xPtoa,yPtoa,xPtob,yPtob);
	}
	/** Seters estáticos para cambiar los parámetros de comportamiento
	 *  de los Boids*/
	
	static public void setPesoLider(double peso){
		pesoLider = peso;
	}
	
	static public void setCohesion(double cohesion){
		pesoCohesion = cohesion;
	}
	
	static public void setSeparacion(double separacion){
		pesoSeparacion = separacion;
	}
	
	static public void setAlineacion(double alineacion){
		pesoAlineacion = alineacion;
	}
	
	static public void setVelObjetivo(double objetivo){
		pesoObjetivo = objetivo;
	}
	
	static public void setEvitaObstaculo(double evitaObs){
		pesoObstaculo = evitaObs;
	}
	/**Determina la velocidad máxima, en módulo, para todos los boids (es estático)*/	
	static public void setVelMax(double veloMax){
		velMax = veloMax;
	}
	/**Fija la posición del objetivo*/
	static public void setObjetivo(double x,double y){
		objetivo.set(0,0,x);
		objetivo.set(1,0,y);
	}
	
	/**Calcula la distancia euclidea existente entre el Boid y el objetivo*/
	public double getDistObjetivo(){
		 double distancia = (objetivo.minus(this.getPosicion())).norm2();
		return distancia;		
	}
	
	/*Geters de los parametros de los Boids*/
	public static double getPesoLider() {
		return pesoLider;
	}
	
	public static double getPesoAlineacion() {
		return pesoAlineacion;
	}
	public static double getPesoCohesion() {
		return pesoCohesion;
	}
	public static double getPesoObjetivo() {
		return pesoObjetivo;
	}
	public static double getPesoObstaculo() {
		return pesoObstaculo;
	}
	public static double getPesoSeparacion() {
		return pesoSeparacion;
	}
	public static double getVelMax() {
		return velMax;
	}
	public static Matrix getObjetivo(){
		return objetivo;
	}
	public boolean isLider() {
		return lider;
	}
	public void setLider(boolean lider) {
		this.lider = lider;
	}
	

	
	public static void main(String[] args) {
		Vector<Boid> bandada = new Vector<Boid>();
		Vector<Obstaculo> obstaculos = new Vector<Obstaculo>();
		int tamanoBandada = 50;
		double coorObjetivo[] = {800,800};
		Matrix objetivo = new Matrix(coorObjetivo,2);
		for (int j = 0;j<tamanoBandada;j++){
			double posAux[] = {Math.random()*800,Math.random()};
			double velAux[] = {Math.random(),Math.random()};
			Matrix posi = new Matrix(posAux,2);
			Matrix vel = new Matrix(velAux,2);			
			bandada.add(new Boid(posi,vel));
			double posObstaculos[] = {Math.random(),Math.random()*500};
			double velObstaculos[] = {0,0};
			Matrix posiObs = new Matrix(posObstaculos,2);
			Matrix velObs = new Matrix(velObstaculos,2);
			obstaculos.add(new Obstaculo(posiObs,velObs));
		}
		System.out.println("bandada original de " + bandada.size());
//		double posAux[] = {Math.random(),Math.random()};
//		double velAux[] = {Math.random(),Math.random()};
//		Matrix posi = new Matrix(posAux,2);
//		Matrix vel = new Matrix(velAux,2);	
		JApplet muestraBoid = new JApplet();
		panelMuestraBoid pintor = new panelMuestraBoid();
//		pintor.introducirBoid(new Boid(posi,vel));
		pintor.introducirBandada(bandada);
		muestraBoid.getContentPane().add(pintor);		
		Console.run(muestraBoid,1000,1000);
		pintor.repaint();
		while(true){
			for (int j = 0;j<tamanoBandada;j++){
				bandada.elementAt(j).mover(bandada,obstaculos,j,objetivo);
			}
			pintor.repaint();
			try {
	            Thread.sleep(20);
	        } catch (Exception e) {
	        }
		}
	}	
}

class panelMuestraBoid extends JPanel{
	Vector<Boid> bandadaPintar = new Vector<Boid>();
	public void introducirBoid(Boid b){
		bandadaPintar.add(b);		
	}
	
	public void introducirBandada(Vector<Boid> banda){
		for (int i =0;i<banda.size();i++){
			bandadaPintar.add(banda.elementAt(i));
		}
	}
	
	public void paintComponent(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		super.paintComponent(g2);
		for (int i=0;i<bandadaPintar.size();i++){
			g2.draw(bandadaPintar.elementAt(i).getForma());
			g2.fill(bandadaPintar.elementAt(i).getForma());			
		}
		float ptosX[] = {800,795,805};
		float ptosY[] = {800,805,805};
		GeneralPath triangulo = new GeneralPath(GeneralPath.WIND_NON_ZERO,ptosX.length);
		triangulo.moveTo (ptosX[0], ptosY[0]);

		for (int index = 1; index < ptosX.length; index++) {
		 	 triangulo.lineTo(ptosX[index], ptosY[index]);
		};
		triangulo.closePath();		
		g2.fill(triangulo);
		g2.setColor(Color.red);
		g2.draw(triangulo);
	}
}