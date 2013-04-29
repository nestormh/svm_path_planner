package boids;

import sibtra.predictivo.Coche;
import Jama.Matrix;

public class Simulador3 extends Simulador{
	
	private int contPensar = 0;
	private int incrPensar = 1;
	
	
	public int getContPensar() {
		return contPensar;
	}


	public void setContPensar(int contPensar) {
		this.contPensar = contPensar;
	}


	public int getIncrPensar() {
		return incrPensar;
	}


	public void setIncrPensar(int incrPensar) {
		this.incrPensar = incrPensar;
	}


	public void moverBoids(Coche ModCoche){
		System.out.println("movemos la segunda bandada");
		marcaObstaculosVisibles(getObstaculos());
		int indLider = 0;
		double distMin = Double.POSITIVE_INFINITY;
		boolean liderEncontrado = false;
		setContIteraciones(getContIteraciones()+1);
		//If para controlar la frecuencia a la que se a√±aden boids a la bandada
		if(getContIteraciones() > getContNuevosBoids()){
			for(int g=0;g<3;g++){				
//				double pos[] = {Math.abs(700*Math.random()),Math.abs(500*Math.random())};
//				double pos[] = {getBandada().lastElement().getPosicion().get(0,0)+10*Math.random(),
//						getBandada().lastElement().getPosicion().get(1,0)+10*Math.random()};
//				double pos[] = {posInicial.get(0,0)+Math.random(), posInicial.get(1,0)+Math.random()};
				double pos[] = {ModCoche.getX()+Math.random(), ModCoche.getY()+Math.random()};
				Matrix posi = new Matrix(pos,2);
				double vel[] = {0,0};
				Matrix velo = new Matrix(vel,2);
				double ace[] = {0,0};
				Matrix acel = new Matrix(ace,2);
				//Indicamos en que iteracion se crea el boid para despues calcular
				//su antiguedad
				this.getBandada().add(new Boid(posi,velo,acel,getContIteraciones()));
//				this.getBandada().add(new Boid(posi,velo,acel));
			}
//			double posCentroEscenario[] = {largoEscenario/2,anchoEscenario/2};
//			Matrix posiCentro = new Matrix(posCentroEscenario,2);
//			double vel[] = {0,0};
//			Matrix velo = new Matrix(vel,2);
//			double ace[] = {0,0};
//			Matrix acel = new Matrix(ace,2);
//			this.getBandada().add(new Boid(posiCentro, velo, acel));
			setContNuevosBoids(getContIteraciones() + getIncrNuevosBoids());
		}
		// Iteramos sobre toda la bandada		
		if (this.getBandada().size() != 0){
			System.out.println("TamaÒo actual de la bandada " + this.getBandada().size());
		
			for (int j = 0;j<this.getBandada().size();j++){
				this.getBandada().elementAt(j).setConectado(false);
				this.getBandada().elementAt(j).setAntiguo((double)getContIteraciones());
				this.getBandada().elementAt(j).calculaValoracion();	
				System.out.println("contPensar " + getContPensar()+"contIteraciones " +getContIteraciones());
				if(getContIteraciones() > getContPensar()){
					System.out.println("dentro del FOR");
//					this.getBandada().elementAt(j).calculaMover(this.getBandada()
//						,getObstaculos(),j,Boid.getObjetivo());
					setObstaculosFuturos(getObstaculos()); 
					//calculo el tiempo que el coche tardarÌa en alcanzar este boid
					double t = this.getBandada().elementAt(j).distThisBoid2Point(getModCoche().getX(),getModCoche().getY())/getModCoche().getVelocidad();
					//prediccion de donde van a estar los obst·culos cuando el coche llegue al luga r que ocupa este boid en este instante
					moverObstaculos(t,getObstaculosFuturos());
					this.getBandada().elementAt(j).calculaMover(this.getBandada()
							,getObstaculosFuturos(),j,Boid.getObjetivo());	
//					this.getBandada().elementAt(j).calculaMover(this.getBandada()
//							,getObstaculos(),j,Boid.getObjetivo());	
				}
//				this.getBandada().elementAt(j).mover(this.getBandada()
//						,getObstaculos(),j,Boid.getObjetivo());
				this.getBandada().elementAt(j).mover();
				double dist = this.getBandada().elementAt(j).getDistObjetivo();
				// Deshabilitamos el liderazgo de la iteraci√≥n anterior antes de retirar ning√∫n 
				// de la bandada por cercan√≠a al objetivo				
//				Si est√° lo suficientemente cerca del objetivo lo quitamos de la bandada
				if (dist < distOk){
					this.getBandada().remove(j);//Simplemente lo quito, no guardo lo que hizo
//					this.getBandada().elementAt(j).setNumIteraciones(getContIteraciones());
//					traspasarBoid(j);
//					numBoidsOk++; // Incremento el numero de boids que han llegado al objetivo
				}
				// Buscamos al lider
//				if(j < this.getBandada().size()){
//					if (this.getBandada().elementAt(j).isCaminoLibre()){										
//						if (dist < distMin){
//							distMin = dist;
//							indLider = j;
//							liderEncontrado = true;
//						}
//					}
//				}
									
			}
			if (getContIteraciones() > getContPensar()){
				setContPensar(getContIteraciones() + getIncrPensar());			
			}
			
//			if (indMinAnt<this.getBandada().size())
//				this.getBandada().elementAt(indMinAnt).setLider(false);
//			if (liderEncontrado && (indLider<getBandada().size())){
//				getBandada().elementAt(indLider).setLider(true);
//			}
		}
//		return indLider;				
	}

}
