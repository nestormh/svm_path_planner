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

import sibtra.util.UtilCalculos;

//import com.bruceeckel.swing.Console;

import Jama.Matrix;

public class Boid implements Serializable{
	/**Vector con las componentes de aceleracion del boid*/
	Matrix aceleracion;
	/**Vector con las componentes de velocidad del boid*/
	Matrix velocidad;
	/**Vector con las componentes de posicion del boid*/
	Matrix posicion;
	/**Objeto gráfico que representará al boid*/
	GeneralPath triangulo;
	/**Linea que muestra la dirección del desplazamiento del boid*/
	Line2D lineaDireccion = new Line2D.Double();
	/**Color del boid para diferenciar las distintas generaciones**/
	Color color;
	Vector<Matrix> rutaBoid = new Vector<Matrix>();
	public boolean lider = false;
	public boolean caminoLibre = false;
	
	//------Campos para la gesti�n de la b�squeda a*------------------------
	/**
	 * Indica si el boid est� en la lista openSet, es decir, aquellos nodos que todav�a pueden ser explorados
	 */
	public boolean openSet = false;	
	/**
	 * Indica si el boid est� en la lista closedSet, es decir, aquellos nodos que ya han sido explorados
	 */
	public boolean closedSet = false;
	/**
	 * Funci�n de costo del camino, que es el costo desde el nodo inicial hasta este
	 */
	public double g_score = 0;	
	/**
	 * Valor de una heur�stica admisible de la distancia hasta el objetivo. En principio en este caso ser� la 
	 * distancia eucl�dea desde este boid hasta el objetivo .ver el m�todo getDistObjetivo() de esta clase
	 */
	public double h_score = 0;
	/**
	 * Valoraci�n completa del boid. Es la suma de g_score + h_score
	 */
	public double f_score = 0;
	/**
	 * �ndice del vecino a trav�s del cual el camino a llegado a este boid
	 */
	public Boid came_from;
	//------Acaban los campos para la gesti�n de la b�squeda a*------------------------		
	public boolean conectado = false;
	static double radioObstaculo = 5.5;//5.5;
//	static double radioObstaculoLejos = 3;
//	static double radioObstaculoCerca = 5;
	static double radioCohesion = 5;
	static double radioSeparacion = 1;//2;
	static double radioAlineacion = 5;//3;
	static double pesoCohesion = 0;//0.1;//0;//0.01;
	static double pesoSeparacion = 1;//0.2;//0.3; // 10
	static double pesoAlineacion = 0;//5; //0.5
	static double pesoObjetivo = 1;  //1
	static double pesoObstaculo = 1;//0.9;//1.2;
	static double pesoCompensacionLateral = 1;//1.6;
//	static double pesoObstaculoCerca = 1;
	static double pesoLider = 0;
	static double velMax = 1.5;
	static double fuerzaMax = 4.5;
	static double masa = 5;
	
	static double pesoDistOrigen = 10000;
	static double pesoAntiguo = 1;
	
	static double coorObjetivo[] = {800,800};
	static Matrix objetivo = new Matrix(coorObjetivo,2);
	static Matrix posInicial;

	/** Dependiendo del valor de esta variable el boid tendrá tendencia a esquivar un obstáculo
	 *  hacia un lado o hacia otro. Este valor se asignará aleatoriamente al crear el objeto 
	 *  Boid*/
	double tendenciaRepulsion = 0;
	/** Indica cuantas iteraciones a tardado el boid en alcanzar el objetivo*/
	int numIteraciones;
	/** Longitud de la ruta seguida por le boid*/
	double longitudRuta;
	/**Iteración del bucle principal en la que se creo el boid*/
	double fechaNacimiento;
	private double valoracion;
	private double experiencia;	
	private double antiguo;

	public double getAntiguo() {
		return antiguo;
	}

	public void setAntiguo(double antiguo) {
		if (antiguo != 0)
			this.antiguo = (antiguo-getFechaNacimiento())/antiguo;
	}

	public double getValoracion() {
		return valoracion;
	}

	public void setValoracion(double valoracion) {
		this.valoracion = valoracion;
	}
	
	public double getExperiencia() {
		return experiencia;
	}

	public void setExperiencia(double experiencia) {
//		this.experiencia = this.experiencia + experiencia ;
		this.experiencia = experiencia ;
	}

	/**Constructor donde se inicializa la posición y velocidad de cada boid,
	 * además de el objeto gráfico que lo representará*/
	public Boid(Matrix posicion, Matrix velocidad,Matrix aceleracion) {
		this.aceleracion = aceleracion;
		this.velocidad = velocidad;
		this.posicion = posicion;
//		this.posInicial = posicion;
//		lineaDireccion.setLine(this.posicion.get(0,0),this.posicion.get(1,0),
//				this.velocidad.get(0,0),this.velocidad.get(1, 0));
		/**Inicialización del aspecto gráfico del cuerpo del boid*/
//		float ptosX[] = {0,-2,2};
//		float ptosY[] = {1,-1,-1};
//		triangulo = new GeneralPath(GeneralPath.WIND_NON_ZERO,ptosX.length);
//		triangulo.moveTo (ptosX[0], ptosY[0]);
//
//		for (int index = 1; index < ptosX.length; index++) {
//		 	 triangulo.lineTo(ptosX[index], ptosY[index]);
//		};
//		triangulo.closePath();
//		triangulo.transform(AffineTransform.getTranslateInstance(posicion.get(0,0),posicion.get(1,0)));
//		this.nuevoPuntoRuta(this.getPosicion());
		tendenciaRepulsion = Math.random();
//		if (aleatorio < 0.5){
//			tendenciaRepulsion = true;			
//		}
//		else
//			tendenciaRepulsion = false;
	}
	
	public Boid(Matrix posicion, Matrix velocidad,Matrix aceleracion,double fechaNacimiento) {
		this(posicion,velocidad,aceleracion);
		setFechaNacimiento(fechaNacimiento);		
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
		int contAlineacion = 0;
		double dist = 0;
		// Bucle que recorre toda la bandada
		for (int i=0;i < bandada.size();i++){
			if (i != indBoid){
				dist = bandada.elementAt(i).getPosicion().minus(this.getPosicion()).norm2();
				if (dist < radioAlineacion){
					velMedia = velMedia.plus(bandada.elementAt(i).getVelocidad());
					contAlineacion++;
				}
				// if para la alineacion
				if (!this.isLider() && !liderCerca){
					if (dist < radioCohesion){
						if (bandada.elementAt(i).isLider()){
							liderCerca = true;
							indLider = i;
						}
						else{
							centroMasa = centroMasa.plus(bandada.elementAt(i).getPosicion());							
							cont++;
						}
					}
				} // if (!this.isLider()) para la cohesion
				if (dist < radioSeparacion){
					separa = separa.minus(bandada.elementAt(i).getPosicion().minus(this.getPosicion()));
					// Los boids más cercanos tienen que producir más repulsión
					if (dist != 0)
						separa = separa.times(1/(dist)*(dist));
//					separa = separa.times(1/(dist)*(dist));
				}// if para la separacion
			} // if (i != indBoid) para todas las reglas
		} // for principal
		// calculos para la separacion
		
		separa = separa.times(pesoSeparacion);
		separa = separa.minus(this.getVelocidad());
		separa = limitaFuerza(separa);
//		separa = separa.times(1/separa.norm2());// lo pasamos a unitario
		// calculos para la velocidad de alineación
		if (contAlineacion != 0){
			velMedia = velMedia.timesEquals((double)1/(double)contAlineacion);
			velMedia = velMedia.minus(this.getVelocidad());			
			velMedia = velMedia.times(pesoAlineacion);
//			velMedia = velMedia.minus(this.getVelocidad());
			velMedia = limitaFuerza(velMedia);
//			velMedia = velMedia.times(1/velMedia.norm2()); // lo pasamos a unitario
		}else{
			velMedia.set(0,0,0);
			velMedia.set(1,0,0);
		}		
		// calculos para la cohesión
		if (cont != 0 && liderCerca == false){
			centroMasa = centroMasa.times((double)1/(double)cont);
			velCohesion = (centroMasa.minus(this.getPosicion())).times(pesoCohesion);
//			velCohesion = (centroMasa.minus(this.getPosicion()));
//			velCohesion = velCohesion.times(pesoCohesion);			
		}
		if(liderCerca == true){
			velCohesion = (bandada.elementAt(indLider).getPosicion().minus(this.getPosicion())).times(pesoLider);
		}
		velCohesion = velCohesion.minus(this.getVelocidad());
		velCohesion = limitaFuerza(velCohesion);
//		velCohesion = velCohesion.times(1/velCohesion.norm2());// lo pasamos a unitario
		velResultante = (velMedia.plus(velCohesion)).plus(separa);
//		velResultante = velResultante.times(1/velResultante.norm2());
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
					if (Math.abs(bandada.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()) < radioCohesion){
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
	/** Calcula el centro de masas alrededor de un boid con un determinado
	 * radio de vecindad*/
	public Matrix calculaCentroMasas(Vector<Boid> bandada,double radioCentroMasas){
		double pos[] = {0,0};
		int cont = 0;
		Matrix centroMasa = new Matrix(pos,2);		
		for (int i=0;i < bandada.size();i++){
				if (Math.abs(bandada.elementAt(i).getPosicion().minus(this.getPosicion()).norm2()) < radioCentroMasas){
					centroMasa = centroMasa.plus(bandada.elementAt(i).getPosicion());					
					cont++;
			}
		}
//		System.out.println("Antes de dividir");
//		centroMasa.print(10,4);
		if (cont != 0){ //Puede que no haya boids en el radio de vecindad
//			centroMasa = centroMasa.timesEquals(1/cont);
			centroMasa.timesEquals((double)1/(double)cont);
//			System.out.println("Después de dividir y cont vale " +cont);
//			centroMasa.print(10,4);
		}
		else{ // si no hay vecinos nos quedamos con la posición del boid
			centroMasa = this.getPosicion();
//			System.out.println("No había vecinos");
//			centroMasa.print(10,4);
		}
		
		return centroMasa;
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
		velObj = velObj.minus(this.getVelocidad());
		velObj = limitaFuerza(velObj);
//		velObj = velObj.times(1/velObj.norm2()); // lo hacemos unitario
		return velObj;
	}
	
	/** Regla para no permitir grandes velocidades*/ 
	
	public Matrix limitaVelocidad(Matrix vel){
		Matrix velLimitada = new Matrix(2,1);
		velLimitada = vel;
		if (Math.abs(vel.norm2()) > velMax)
			velLimitada = vel.times(1/vel.norm2()).times(velMax);
		return velLimitada;
	}
	
/** Regla para no permitir grandes aceleraciones*/ 
	
	public Matrix limitaFuerza(Matrix fuerza){
		Matrix fuerzaLimitada = new Matrix(2,1);
		fuerzaLimitada = fuerza;
		if (Math.abs(fuerza.norm2()) > fuerzaMax)
			fuerzaLimitada = fuerza.times(1/fuerza.norm2()).times(fuerzaMax);
		return fuerzaLimitada;
	}

/** Regla para no permitir grandes aceleraciones*/ 
	
	public Matrix limitaFuerza(Matrix fuerza,double fuerzaMaxima){
		Matrix fuerzaLimitada = new Matrix(2,1);
		fuerzaLimitada = fuerza;
		if (Math.abs(fuerza.norm2()) > fuerzaMaxima)
			fuerzaLimitada = fuerza.times(1/fuerza.norm2()).times(fuerzaMaxima);
		return fuerzaLimitada;
	}

	/** Regla para esquivar los obstáculos*/
	
	public Matrix evitaObstaculo(Vector<Obstaculo> obstaculos,Boid b,Matrix direcObjetivo){
		double pos[] = {0,0};
		double zero[] = {0,0};
		Matrix cero = new Matrix(zero,2);
		Matrix c = new Matrix(pos,2);
		Matrix repulsion = new Matrix(zero,2);
		Matrix direcBoidObstaculo = new Matrix(zero,2);
		Matrix compensacion = new Matrix(zero,2);
		boolean caminoOcupado = false;
		double dist = 0;
		double umbralEsquivar = Math.toRadians(20);//20//5
		double umbralCaso3 = -Math.toRadians(10);//10
		int sentidoCompensacionLateral = 0;
		Line2D recta = 
			new Line2D.Double(this.getPosicion().get(0,0),this.getPosicion().get(1,0)
						,Boid.getObjetivo().get(0,0),Boid.getObjetivo().get(1,0));
		for (int i=0;i < obstaculos.size();i++){
			dist = obstaculos.elementAt(i).getPosicion().minus(this.getPosicion()).norm2();			
			if (dist < radioObstaculo){
//			if (dist < radioObstaculo*(1+this.getDistOrigen()/100)){
				repulsion = repulsion.minus(obstaculos.elementAt(i).getPosicion().minus(this.getPosicion()));
				//es el vector que apunta desde al boid hacia el obstáculo
				if (dist != 0){ // Para no dividir entre cero
					repulsion = repulsion.times(1/(dist)*(dist));
				}
				repulsion = repulsion.times(pesoObstaculo);
				//Dependiendo de la velocidad del obstáculo, de la posición del Boid
				//y de la posición del objetivo, se calculará una compensación lateral
				direcBoidObstaculo = repulsion.times(-1);
				double angVelObst = Math.atan2(obstaculos.elementAt(i).getVelocidad().get(1,0),
						obstaculos.elementAt(i).getVelocidad().get(0,0));
				double angDirecBoidObstaculo = Math.atan2(direcBoidObstaculo.get(1,0),
						direcBoidObstaculo.get(0,0));
				double angDirecObjetivo = Math.atan2(direcObjetivo.get(1,0),
						direcObjetivo.get(0, 0));
				double angCompensacion = 0;
				// Solo producen repulsión aquellos obstáculos que se encuentren entre el objetivo
				// y el boid, los que quedan detrás del boid no influencian
				if (UtilCalculos.diferenciaAngulos(angDirecObjetivo, angDirecBoidObstaculo)< 3*Math.PI/2){
					//Diferencia entre el ángulo formado por el vector desde el boid hacia
					//el obstáculo y la velocidad del obstáculo y el ángulo formado entre
					//el vector que va desde el boid hacia el objetivo y la velocidad del
					//obstáculo
					double angObsBoidObj = UtilCalculos.diferenciaAngulos(angVelObst,angDirecBoidObstaculo) -
							UtilCalculos.diferenciaAngulos(angVelObst, angDirecObjetivo);
					// caso en el que el boid y el obstáculo van a cruzar sus caminos 
					// en el futuro
//					if (UtilCalculos.diferenciaAngulos(angVelObst,angDirecBoidObstaculo) >=
//						UtilCalculos.diferenciaAngulos(angVelObst, angDirecObjetivo)){
					if (angObsBoidObj >= umbralCaso3){
//						if (UtilCalculos.diferenciaAngulos(angDirecBoidObstaculo, angDirecObjetivo) <= umbralEsquivar){
						if (angObsBoidObj > umbralEsquivar){// Por delante
//							System.out.println("va por delante del  obstáculo");
							compensacion.set(0,0,repulsion.get(1,0));
							compensacion.set(1,0,-repulsion.get(0,0));
							angCompensacion = Math.atan2(compensacion.get(1,0),
									compensacion.get(0,0));
							if (UtilCalculos.
									diferenciaAngulos(angVelObst,angCompensacion)>Math.toRadians(90)){
								//Si se da la condición lo cambiamos de sentido, si no se queda 
								//como se calculó antes del if
								compensacion.set(0,0,-repulsion.get(1,0));
								compensacion.set(1,0,repulsion.get(0,0));
							}
							
//							sentidoCompensacionLateral = -1;
						}else{
							//Por detrás 
//							System.out.println("va por detrás del  obstáculo");
							compensacion.set(0,0,repulsion.get(1,0));
							compensacion.set(1,0,-repulsion.get(0,0));
							angCompensacion = Math.atan2(compensacion.get(1,0),
									compensacion.get(0,0));
							if (UtilCalculos.
									diferenciaAngulos(angVelObst,angCompensacion)<Math.toRadians(90)){
								//Si se da la condición lo cambiamos de sentido, si no se queda 
								//como se calculó antes del if
								compensacion.set(0,0,-repulsion.get(1,0));
								compensacion.set(1,0,repulsion.get(0,0));
							}
//							sentidoCompensacionLateral = 1;
						}
//						if (!(pesoCompensacionLateral == 0)){
						compensacion.timesEquals(pesoCompensacionLateral);
							//						System.out.println("Calculamos compensación lejos");
//							compensacion = obstaculos.elementAt(i).getVelocidad().times(
//									pesoCompensacionLateral*sentidoCompensacionLateral);
//						}
						c = c.plus(repulsion.plus(compensacion));
					}else{//Si no va a cruzarse con el obstáculo no se le añade compensación lateral
						//ni repulsion
											c = c.plus(repulsion);
//						c = c.plus(cero);
					}
				}				
				
			}
//			if ((dist > radioObstaculoCerca) && (dist < radioObstaculoLejos)){
//				repulsion = repulsion.minus(obstaculos.elementAt(i).getPosicion().minus(this.getPosicion()));
//				if (dist != 0){
//					repulsion = repulsion.times(1/(dist)*(dist));
//				}
//				repulsion = repulsion.times(pesoObstaculo);
//				if (!(pesoObstaculoLejos == 0)){
////					System.out.println("Calculamos compensación lejos");
//					compensacion = obstaculos.elementAt(i).getVelocidad().times(pesoObstaculoLejos);
//				}				
//				c = c.plus(repulsion.plus(compensacion));
//			}
//			else if(dist < radioObstaculoCerca){
//				repulsion = repulsion.minus(obstaculos.elementAt(i).getPosicion().minus(this.getPosicion()));				
//				if (dist != 0){
//					repulsion = repulsion.times(1/(dist)*(dist));
//				}
//				repulsion = repulsion.times(pesoObstaculo);
//				if (!(pesoObstaculoCerca == 0)){
////					System.out.println("Calculamos compensación cerca");
//					compensacion = obstaculos.elementAt(i).getVelocidad().times(-pesoObstaculoCerca);
//				}
//				c = c.plus(repulsion.plus(compensacion));
//			}
		
			if (!caminoOcupado)// Sólo se calcula la intersección mientras el camino siga sin ocupar
				caminoOcupado = recta.intersects(obstaculos.elementAt(i).getForma());
		}

		setCaminoLibre(!caminoOcupado);// Si no tiene el camino ocupado por el momento es el lider
		 
		c = c.minus(this.getVelocidad());
//		c = limitaFuerza(c);
//		c = c.times(1/c.norm2());// lo hacemos unitario
		return c;
	}
	
	/** Método que calcula todas las reglas para cada Boid, las suma vectorialmente
	 * 	, calcula el desplazamiento y lo realiza*/
	
	public void calculaMover(Vector<Boid> bandada,Vector<Obstaculo> obstaculos,int indBoid, Matrix obj){
		Matrix desp = new Matrix(2,1);
//		Matrix despCohesion = new Matrix(2,1);
//		Matrix despSeparacion = new Matrix(2,1);
//		Matrix despAlineacion = new Matrix(2,1);
		Matrix despObjetivo = new Matrix(2,1);
		Matrix despAliCoheSep = new Matrix(2,1);
		Matrix despObstaculo = new Matrix(2,1);
		Matrix haciaObjetivo = new Matrix(2,1);
		haciaObjetivo = objetivo.minus(this.getPosicion());
//		System.out.println("el objetivo est� en "+objetivo.get(0, 0)+" "+objetivo.get(1, 0));
//		despCohesion = cohesion(bandada, indBoid);
//		despSeparacion = separacion(bandada, indBoid);
//		despAlineacion = alineacion(bandada, indBoid);
		despAliCoheSep = aliCoheSep(bandada, indBoid);
		despObjetivo = seguirObjetivo(bandada,indBoid,obj);
		despObstaculo = evitaObstaculo(obstaculos,bandada.elementAt(indBoid),haciaObjetivo);
//		desp = ((despAliCoheSep.plus(despObjetivo)).plus(despObstaculo)).plus(this.getVelocidad());
		desp = ((despAliCoheSep.plus(despObjetivo)).plus(despObstaculo));
//		desp = desp.times(1/desp.norm2());
//		desp = limitaVelocidad(((despAliCoheSep.plus(despObjetivo)).plus(despObstaculo)).plus(this.getVelocidad()));
		desp = desp.timesEquals(1/masa); // Simula la masa del boid
		desp = limitaFuerza(desp);
		setAceleracion(desp);
//		desp = limitaVelocidad(despCohesion.plus(despSeparacion).plus(despAlineacion).plus(despObjetivo).plus(despObstaculo).plus(this.getVelocidad()));
//		this.getForma().transform(AffineTransform.getTranslateInstance(desp.get(0,0), desp.get(1,0)));
//		this.setVelocidad(desp);
		this.setVelocidad(limitaVelocidad(getAceleracion().plus(this.getVelocidad()))); 
//		this.setPosicion(this.getPosicion().plus(this.getVelocidad()));
//		setLineaDireccion(getPosicion().get(0,0),getPosicion().get(1,0),
//				(getPosicion().plus(getVelocidad().times(2))).get(0,0),
//				(getPosicion().plus(getVelocidad().times(2))).get(1,0));
	}
	
	public void mover(){
//		this.setVelocidad(limitaVelocidad(getAceleracion().plus(this.getVelocidad()))); 
		this.setPosicion(this.getPosicion().plus(this.getVelocidad()));
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
	 
	public double  calculaLongRuta(){
		longitudRuta = 0;
		for(int i=0;i<rutaBoid.size()-1;i++){
			double dx=rutaBoid.elementAt(i+1).get(0,0)-rutaBoid.elementAt(i).get(0,0);
			double dy=rutaBoid.elementAt(i+1).get(1,0)-rutaBoid.elementAt(i).get(1,0);
			longitudRuta = longitudRuta + Math.sqrt(dx*dx + dy*dy);
		}
		return longitudRuta;
	}
	
	public double getLongitudRuta() {
		return longitudRuta;
	}

	public void setLongitudRuta(double longitudRuta) {
		this.longitudRuta = longitudRuta;
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
	public Matrix getAceleracion() {
		return aceleracion;
	}

	public void setAceleracion(Matrix aceleracion) {
		this.aceleracion = aceleracion;
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
	
	static public void setPesoCohesion(double cohesion){
		pesoCohesion = cohesion;
	}
	
	static public void setPesoSeparacion(double separacion){
		pesoSeparacion = separacion;
	}
	
	static public void setPesoAlineacion(double alineacion){
		pesoAlineacion = alineacion;
	}
	
	static public void setPesoObjetivo(double objetivo){
		pesoObjetivo = objetivo;
	}
	
	static public void setPesoObstaculo(double evitaObs){
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
	
	public double getFechaNacimiento() {
		return fechaNacimiento;
	}

	public void setFechaNacimiento(double fechaNacimiento) {
		this.fechaNacimiento = fechaNacimiento;
	}

	/**
	 * 
	 * @return Posición inicial de la bandada
	 */
	public static Matrix getPosInicial() {
		return posInicial;
	}
	/**
	 * Setea la posición inicial de la bandada
	 * @param posInicial
	 */
	public static void setPosInicial(Matrix posInicial) {
		Boid.posInicial = posInicial;
	}
	/**Calcula la distancia euclidea existente entre el Boid y la posición inicial de la bandada*/
	public double getDistOrigen(){
		 double distancia = (posInicial.minus(this.getPosicion())).norm2();
		return distancia;		
	}
	
	public void calculaValoracion(){
//		valoracion = (pesoDistOrigen/getDistOrigen()) + (pesoAntiguo/getAntiguo() + getExperiencia());
		valoracion = (pesoDistOrigen/getDistObjetivo());//+(pesoAntiguo/getAntiguo());
//		valoracion = (pesoDistOrigen/getDistOrigen())*(getAntiguo());
//		valoracion = (pesoDistOrigen/getDistObjetivo()) + getExperiencia();
//		valoracion = pesoDistOrigen/getDistOrigen();
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
//	public static double getRadioObstaculoLejos() {
//		return radioObstaculoLejos;
//	}
//
//	public static void setRadioObstaculoLejos(double radioObstaculoLejos) {
//		Boid.radioObstaculoLejos = radioObstaculoLejos;
//	}
//
//	public static double getRadioObstaculoCerca() {
//		return radioObstaculoCerca;
//	}
//
//	public static void setRadioObstaculoCerca(double radioObstaculoCerca) {
//		Boid.radioObstaculoCerca = radioObstaculoCerca;
//	}

	public static double getPesoCompensacionLateral() {
		return pesoCompensacionLateral;
	}

	public static void setPesoCompensacionLateral(double pesoCompensacionLateral) {
		Boid.pesoCompensacionLateral = pesoCompensacionLateral;
	}

//	public static double getPesoObstaculoCerca() {
//		return pesoObstaculoCerca;
//	}
//
//	public static void setPesoObstaculoCerca(double pesoObstaculoCerca) {
//		Boid.pesoObstaculoCerca = pesoObstaculoCerca;
//	}

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
	public boolean isConectado() {
		return conectado;
	}

	public void setConectado(boolean conectado) {
		this.conectado = conectado;
	}
	
	public boolean isOpenSet() {
		return openSet;
	}

	public void setOpenSet(boolean openSet) {
		this.openSet = openSet;
	}

	public boolean isClosedSet() {
		return closedSet;
	}

	public void setClosedSet(boolean closedSet) {
		this.closedSet = closedSet;
	}


	
	public static void main(String[] args) {
//		Vector<Boid> bandada = new Vector<Boid>();
//		Vector<Obstaculo> obstaculos = new Vector<Obstaculo>();
//		int tamanoBandada = 50;
//		double coorObjetivo[] = {800,800};
//		Matrix objetivo = new Matrix(coorObjetivo,2);
//		for (int j = 0;j<tamanoBandada;j++){
//			double posAux[] = {Math.random()*800,Math.random()};
//			double velAux[] = {Math.random(),Math.random()};
//			Matrix posi = new Matrix(posAux,2);
//			Matrix vel = new Matrix(velAux,2);			
//			bandada.add(new Boid(posi,vel));
//			double posObstaculos[] = {Math.random(),Math.random()*500};
//			double velObstaculos[] = {0,0};
//			Matrix posiObs = new Matrix(posObstaculos,2);
//			Matrix velObs = new Matrix(velObstaculos,2);
//			obstaculos.add(new Obstaculo(posiObs,velObs));
//		}
//		System.out.println("bandada original de " + bandada.size());
//		double posAux[] = {Math.random(),Math.random()};
//		double velAux[] = {Math.random(),Math.random()};
//		Matrix posi = new Matrix(posAux,2);
//		Matrix vel = new Matrix(velAux,2);	
//		JApplet muestraBoid = new JApplet();
//		panelMuestraBoid pintor = new panelMuestraBoid();
//		pintor.introducirBoid(new Boid(posi,vel));
//		pintor.introducirBandada(bandada);
//		muestraBoid.getContentPane().add(pintor);		
//		Console.run(muestraBoid,1000,1000);
//		pintor.repaint();
//		while(true){
//			for (int j = 0;j<tamanoBandada;j++){
//				bandada.elementAt(j).mover(bandada,obstaculos,j,objetivo);
//			}
//			pintor.repaint();
//			try {
//	            Thread.sleep(20);
//	        } catch (Exception e) {
//	        }
//		}
	}

	public boolean isCaminoLibre() {
		return caminoLibre;
	}

	public void setCaminoLibre(boolean caminoLibre) {
		this.caminoLibre = caminoLibre;
	}

	public static double getRadioAlineacion() {
		return radioAlineacion;
	}

	public static void setRadioAlineacion(double radioAlineacion) {
		Boid.radioAlineacion = radioAlineacion;
	}

	public static double getRadioCohesion() {
		return radioCohesion;
	}

	public static void setRadioCohesion(double radioCohesion) {
		Boid.radioCohesion = radioCohesion;
	}

	public static double getRadioSeparacion() {
		return radioSeparacion;
	}

	public static void setRadioSeparacion(double radioSeparacion) {
		Boid.radioSeparacion = radioSeparacion;
	}

	public static double getRadioObstaculo() {
		return radioObstaculo;
	}
	
	public int getNumIteraciones() {
		return numIteraciones;
	}

	public void setNumIteraciones(int numIteraciones) {
		this.numIteraciones = numIteraciones;
	}

	public static void setRadioObstaculo(double radioObstaculo) {
		Boid.radioObstaculo = radioObstaculo;
	}

	public void setRutaBoid(Vector<Matrix> rutaBoid) {
		this.rutaBoid = rutaBoid;
	}	
	
	public double getG_score() {
		return g_score;
	}

	public void setG_score(double g_score) {
		this.g_score = g_score;
	}

	public double getH_score() {
		return h_score;
	}

	public void setH_score(double h_score) {
		this.h_score = h_score;
	}

	public double getF_score() {
		return f_score;
	}

	public void setF_score(double f_score) {
		this.f_score = f_score;
	}
	public double calculaF_score(){
		this.setF_score(this.getG_score() + this.getH_score());
		return this.f_score;
	}

	public Boid getCame_from() {
		return came_from;
	}

	public void setCame_from(Boid came_from) {
		this.came_from = came_from;
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