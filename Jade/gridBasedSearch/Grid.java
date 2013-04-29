package gridBasedSearch;

import java.util.Vector;

import Jama.Matrix;
import boids.Boid;
import boids.Obstaculo;
import boids.Simulador;

public class Grid {
	double[][] matrizPesos = new double[30][30];
	double resolution;
	double longitudX;	
	double longitudY;
	int numPtosX;
	int numPtosY;
	GridSearchPoint[][] rejilla;
	/**
	 * posición del objetivo en términos de índices de la rejilla
	 */
	int[] goalPos;	
	/**
	 * posición de inicio en términos de índices de la rejilla
	 */
	int[] startPos;
	/**
	 * Vector que contiene los obstáculos del escenario. Cada miembro de este vector es un objeto de la clase obstaculo 
	 * donde se almacena su forma, su velocidad, su posición, etc
	 */
	Vector<Obstaculo> obstaculos;
	/**
	 * Vector que contiene los boids presentes en el escenario. Cada miembro es un objeto de la clase Boid, donde entre otras cosas se encuenta
	 * la información de posición, velocidad, etc... de cada boid.
	 */
	Vector<Boid> boids;		
	/**
	 * Clase simulador que usaremos para mover los obstáculos, para hacer la predicción de laas posiciones de los obstáculos
	 * pasado un cierto tiempo. Esta clase tiene todos los métodos necesarios
	 */
	Simulador sim;
	/**
	 * Velocidad del coche, lo usaremos para realizar las predicciones de donde van a estar los obstáculos cuando el coche se 
	 * vaya a cruzar con ellos
	 */
	double velCoche;
	int centro;

	public Grid(double resolution, double longitudX, double longitudY,double radioMax){	
//		sim = new Simulador();
		obstaculos = new Vector<Obstaculo>();
		boids = new Vector<Boid>();
		this.resolution = resolution;
		this.longitudX = longitudX;
		this.longitudY = longitudY;
		setNumPtosX((int)Math.floor(longitudX/resolution));
		setNumPtosY((int)Math.floor(longitudY/resolution));
		//Creo el grid con el número de celdas dado por la resolución y por el tamaño real deseado
		rejilla = new GridSearchPoint[getNumPtosX()][getNumPtosY()];
		//lo relleno con GridSearchPoints
		for (int i=0;i<this.numPtosX;i++){
			for (int j=0;j<this.numPtosY;j++){
				this.rejilla[i][j] = new GridSearchPoint(i, j, this.resolution);
			}
		}
		this.matrizPesos = crearRejillaPesos((int)Math.ceil(radioMax/this.resolution)*10);
	}
	
	public void addObstacle(double posX, double posY, double dimensionX, double dimensionY){
//		boolean gridOccupiedLeft = true;
//		boolean gridOccupiedRight = true;
//		boolean gridOccupiedUp = true;
//		boolean gridOccupiedDown = true;		
		int xIndex = (int)Math.floor(posX/getResolution());
		int yIndex = (int)Math.floor(posY/getResolution());
		int offSetCeldasX = (int)Math.floor(dimensionX/getResolution());
		int offSetCeldasY = (int)Math.floor(dimensionY/getResolution());
		int limitInfX = xIndex - offSetCeldasX;
		int limitSupX = xIndex + offSetCeldasX;
		int limitInfY = yIndex - offSetCeldasY;
		int limitSupY = yIndex + offSetCeldasY;

		if (limitInfX < 0){
			limitInfX = 0;
		}else if(limitInfX > numPtosX-1){
			limitInfX = numPtosX-1;
		}
		
		if (limitInfY < 0){
			limitInfY = 0;
		}else if(limitInfY > numPtosY-1){
			limitInfY = numPtosY-1;
		}
		
		
		if (limitSupX < 0){
			limitSupX = 0;
		}else if(limitSupX > numPtosX-1){
			limitSupX = numPtosX-1;
		}
		
		if (limitSupY < 0){
			limitSupY = 0;
		}else if(limitSupY > numPtosY-1){
			limitSupY = numPtosY-1;
		}
		
		//Marcamos las casillas ocupadas por obstáculos
		for (int i=limitInfX;i<limitSupX;i++){
			if (i < 0 || i > this.numPtosX-1){//comprobamos que no sobrepasamos los límites de la rejilla
//				System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
				continue;
			}
			for (int j=limitInfY;j<limitSupY;j++){
				if (j < 0 || j > this.numPtosY-1){//comprobamos que no sobrepasamos los límites de la rejilla
//					System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
					continue;
				}
				this.getRejilla()[i][j].setOccupied(true);
//				System.out.println("casilla "+i+","+j+ " ocupada");
			}
		}
	}
	
	/**
	 * Añade un grupo de obstáculos
	 * @param obstaculos Vector de obstáculos
	 */
	
	public void addObstacles(Vector<Obstaculo> obstaculos){
		for (int i=0; i < obstaculos.size();i++){
			double posXObs = obstaculos.elementAt(i).getPosicion().get(0,0);
			double posYObs = obstaculos.elementAt(i).getPosicion().get(1,0);
			double dimensionX = obstaculos.elementAt(i).getLado()/2;
			double dimensionY = obstaculos.elementAt(i).getLado()/2;
//			System.out.println("creamos la rejilla ");
			this.addObstacle(posXObs, posYObs, dimensionX, dimensionY);
		}
	}
	
	public double[][] crearRejillaPesos(int numCeldasLado){
		centro = (int) Math.ceil(numCeldasLado/2);
		double[][] rejillaPesos = new double[numCeldasLado][numCeldasLado];
		for (int i=0; i < numCeldasLado; i++){
			for (int j=0; j < numCeldasLado; j++){
				rejillaPesos[i][j] = Math.sqrt(Math.pow((i+1-centro), 2)+Math.pow((j+1-centro), 2));				
			}
		}
		this.matrizPesos = rejillaPesos;
		return rejillaPesos;
	}
	
	/**
	 * Marca el entorno del boid como libre y añade un peso a cada celda, dependiendo de la distancia hasta el boid
	 * @param posX
	 * @param posY
	 * @param dimensionX
	 * @param dimensionY
	 */
	public void addBoid(double posX, double posY, double radioBusqueda){
//		boolean gridOccupiedLeft = true;
//		boolean gridOccupiedRight = true;
//		boolean gridOccupiedUp = true;
//		boolean gridOccupiedDown = true;		
		int xIndexBoid = (int)Math.floor(posX/getResolution());
		int yIndexBoid = (int)Math.floor(posY/getResolution());
		int offSetCeldas = (int)Math.floor(radioBusqueda/getResolution());
		int limitInfX = xIndexBoid - offSetCeldas;
		int limitSupX = xIndexBoid + offSetCeldas;
		int limitInfY = yIndexBoid - offSetCeldas;
		int limitSupY = yIndexBoid + offSetCeldas;

		if (limitInfX < 0){
			limitInfX = 0;
		}else if(limitInfX > numPtosX-1){
			limitInfX = numPtosX-1;
		}
		
		if (limitInfY < 0){
			limitInfY = 0;
		}else if(limitInfY > numPtosY-1){
			limitInfY = numPtosY-1;
		}
		
		
		if (limitSupX < 0){
			limitSupX = 0;
		}else if(limitSupX > numPtosX-1){
			limitSupX = numPtosX-1;
		}
		
		if (limitSupY < 0){
			limitSupY = 0;
		}else if(limitSupY > numPtosY-1){
			limitSupY = numPtosY-1;
		}
		
		//Marcamos las casillas cercanas a los boids como libres y les asignamos un peso
		for (int i=limitInfX;i<limitSupX;i++){
			if (i < 0 || i > this.numPtosX-1){//comprobamos que no sobrepasamos los límites de la rejilla
//				System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
				continue;
			}
			for (int j=limitInfY;j<limitSupY;j++){
				if (j < 0 || j > this.numPtosY-1){//comprobamos que no sobrepasamos los límites de la rejilla
//					System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
					continue;
				}			
				//Se marca la celda como libre
				this.getRejilla()[i][j].setOccupied(false);
//				System.out.println("Celda "+i+","+j+" libre");
				// Se asigna un peso de marcado igual a la distancia con el boid más cercano, por eso se comprueba si la distancia con el
				// boid actual es menor que el peso de marcado que la celda tenga asignado, de manera que sólo se sobreescribe si la distancia
				// con el boid actual es menor. Antes de añadir los boids a la rejilla con este método, la rejilla tiene que haber sido reseteada
				// con el método marcarTodoOcupado, que además de marcar todas las celdas como ocupadas asigna un valor infinito a el peso de marcado de 
				//cada celda
				double gananciaPesado = 3;
				double peso_marcado = this.matrizPesos[i-xIndexBoid+centro][j-yIndexBoid+centro]*getResolution()*gananciaPesado; 
				if(peso_marcado < this.getRejilla()[i][j].getPeso_marcado()){
					this.getRejilla()[i][j].setPeso_marcado(peso_marcado);
				}			
//				System.out.println("casilla "+i+","+j+ " marcada como libre con un peso = " + this.getRejilla()[i][j].getPeso_marcado());
			}
		}
	}
	
	/**
	 * Método que comprueba la existencia de partículas en torno a una celda de la rejilla. 
	 * @param indiceX Índice x de la celda en estudio. No puede ser menor que cero ni mayor que el número máximo de celdas en x
	 * @param indiceY Índice y de la celda en estudio. No puede ser menor que cero ni mayor que el número máximo de celdas en y
	 * @param offSetCeldas Distancia en celdas hasta la que se quiere estudiar la existencia de partículas vecinas
	 * @return Vector de Boid que contiene las partículas vecinas que se encuentran a una distancia menor al offSetCeldas
	 */
	public Vector<Boid> buscarParticulasVecinas(int indiceX,int indiceY,int offSetCeldas){
		Vector<Boid> particulasVecinas = new Vector<Boid>();
		particulasVecinas.clear();
		if (indiceX < 0 || indiceX > numPtosX-1 || indiceY < 0 || indiceY > numPtosY-1){
//			System.out.println("la partícula está fuera del escenario");
			return particulasVecinas;
		}
		int limitInfX = indiceX - offSetCeldas;
		int limitSupX = indiceX + offSetCeldas;
		int limitInfY = indiceY - offSetCeldas;
		int limitSupY = indiceY + offSetCeldas;
		
		if (limitInfX < 0){
			limitInfX = 0;
		}else if(limitInfX > numPtosX-1){
			limitInfX = numPtosX-1;
		}
		
		if (limitInfY < 0){
			limitInfY = 0;
		}else if(limitInfY > numPtosY-1){
			limitInfY = numPtosY-1;
		}
		
		
		if (limitSupX < 0){
			limitSupX = 0;
		}else if(limitSupX > numPtosX-1){
			limitSupX = numPtosX-1;
		}
		
		if (limitSupY < 0){
			limitSupY = 0;
		}else if(limitSupY > numPtosY-1){
			limitSupY = numPtosY-1;
		}
		
		for (int i=limitInfX;i<limitSupX;i++){
			if (i < 0 || i > this.numPtosX-1){//comprobamos que no sobrepasamos los límites de la rejilla
//				System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
				continue;
			}
			for (int j=limitInfY;j<limitSupY;j++){
				if (j < 0 || j > this.numPtosY-1){//comprobamos que no sobrepasamos los límites de la rejilla
//					System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
					continue;
				}			
				if (this.getRejilla()[i][j].getParticulas().isEmpty()){
					continue;
				}else{
					for (int k=0;k<this.getRejilla()[i][j].getParticulas().size();k++){
						particulasVecinas.add(this.getRejilla()[i][j].getParticulas().elementAt(k));
					}
					
				}
			}
		}
		return particulasVecinas;
	}
	
	public Vector<Obstaculo> buscarObstaculosVecinos(int indiceX,int indiceY,int offSetCeldas){
		Vector<Obstaculo> obstaculosVecinos = new Vector<Obstaculo>();
		obstaculosVecinos.clear();
		if (indiceX < 0 || indiceX > numPtosX-1 || indiceY < 0 || indiceY > numPtosY-1){
//			System.out.println("El obstáculo está fuera del escenario");
			return obstaculosVecinos;
		}
		int limitInfX = indiceX - offSetCeldas;
		int limitSupX = indiceX + offSetCeldas;
		int limitInfY = indiceY - offSetCeldas;
		int limitSupY = indiceY + offSetCeldas;
		
		if (limitInfX < 0){
			limitInfX = 0;
		}else if(limitInfX > numPtosX-1){
			limitInfX = numPtosX-1;
		}
		
		if (limitInfY < 0){
			limitInfY = 0;
		}else if(limitInfY > numPtosY-1){
			limitInfY = numPtosY-1;
		}
		
		
		if (limitSupX < 0){
			limitSupX = 0;
		}else if(limitSupX > numPtosX-1){
			limitSupX = numPtosX-1;
		}
		
		if (limitSupY < 0){
			limitSupY = 0;
		}else if(limitSupY > numPtosY-1){
			limitSupY = numPtosY-1;
		}
		for (int i=limitInfX;i<limitSupX;i++){
			if (i < 0 || i > this.numPtosX-1){//comprobamos que no sobrepasamos los límites de la rejilla
//				System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
				continue;
			}
			for (int j=limitInfY;j<limitSupY;j++){
				if (j < 0 || j > this.numPtosY-1){//comprobamos que no sobrepasamos los límites de la rejilla
//					System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
					continue;
				}			
				if (this.getRejilla()[i][j].getObstaculos().isEmpty()){
					continue;
				}else{
					for (int k=0;k<this.getRejilla()[i][j].getObstaculos().size();k++){
						obstaculosVecinos.add(this.getRejilla()[i][j].getObstaculos().elementAt(k));
					}
					
				}
			}
		}
		return obstaculosVecinos;
	}
	
	/**
	 * Añade un grupo de boids
	 * @param obstaculos Vector de boids
	 */
	
	public void addBoids(Vector<Boid> boids,double RadioBusqueda){		
		for (int i=0; i < boids.size();i++){
			double posXBoid = boids.elementAt(i).getPosicion().get(0,0);
			double posYBoid = boids.elementAt(i).getPosicion().get(1,0);
//			double dimensionX = RadioBusqueda;//RadioBusqueda/2
//			double dimensionY = dimensionX;
//			System.out.println("introducimos los boids en la rejilla ");
			this.addBoid(posXBoid, posYBoid, RadioBusqueda);
		}		
	}
	
	public Vector<Matrix> busquedaAEstrella(){
		
		boolean caminoCompleto = false;
		boolean tentative_is_better = false;		
		double minF_score = Double.POSITIVE_INFINITY;
		int indMin = 0;
		Vector<Matrix> camino = new Vector<Matrix>();
		Vector<GridSearchPoint> openSet = new Vector<GridSearchPoint>();
		Vector<GridSearchPoint> closedSet = new Vector<GridSearchPoint>();		
		openSet.clear();
		closedSet.clear();	
		clearSearchData();//Desmarcamos las celdas marcadas en la iteración anterior como pertenecientes al open o closedset
		int [] start = getStartPos();
		int [] goal = getGoalPos();	
		this.rejilla[start[0]][start[1]].setG_score(0);
		this.rejilla[start[0]][start[1]].setH_score(this.rejilla[start[0]][start[1]].distThisPoint2Point(goal[0],goal[1]));
		this.rejilla[start[0]][start[1]].setF_score(this.rejilla[start[0]][start[1]].getG_score()+
				this.rejilla[start[0]][start[1]].getH_score());
		this.rejilla[start[0]][start[1]].setOpenSet(true);
		openSet.add(this.rejilla[start[0]][start[1]]);
		GridSearchPoint actual = this.rejilla[start[0]][start[1]];		
		while (!openSet.isEmpty()){
//			System.out.println("tamaño del openset "+openSet.size());
//			System.out.println("tamaño del closedSet "+closedSet.size());
			minF_score = Double.POSITIVE_INFINITY;
			//Buscamos el nodo del opneSet con mejor f_score
			for (int i=0;i<openSet.size();i++){
				if(openSet.elementAt(i).getF_score() < minF_score) {
					indMin=i;
					minF_score=openSet.elementAt(i).getF_score();
				}								
			}
			actual = openSet.elementAt(indMin);
//			System.out.println("índices del punto actual "+actual.getxIndex()+" "+actual.getyIndex()+" y su f_score "+actual.getF_score());
			//comprobamos si el nodo actual es el objetivo
			if (Math.abs(actual.getxIndex() - goal[0]) < 2 && Math.abs(actual.getyIndex() - goal[1]) < 2){ //Asegurarse de que la comprobación funciona
//			if (actual.getxIndex() == goal[0] && actual.getyIndex() == goal[1]){ //Asegurarse de que la comprobación funciona
				caminoCompleto = true;
//				System.out.println("se completó con éxito el camino, supuestamente");
				return reconstruirCaminoAEstrella(actual);
				
			}
			//Quitamos el nodo actual del openSet 
			openSet.remove(indMin); //También existe un método para quitar un elemento especificando que objeto hay que quitar
			//y lo añadimos al closedSet
			actual.setOpenSet(false);
			actual.setClosedSet(true);
			closedSet.add(actual);
//			System.out.println("tamaño de la rejilla "+this.getNumPtosX()+" "+this.getNumPtosY());
			//Exploramos los vecinos del actual
			for (int j=actual.getxIndex()-1;j<=actual.getxIndex()+1;j++){
//				System.out.println("posición de goal "+goal[0]+" "+goal[1]);
				if (j < 0 || j > this.numPtosX-1){//comprobamos que no sobrepasamos los límites de la rejilla
//					System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
					continue;
				}				
				for (int k=actual.getyIndex()-1;k<=actual.getyIndex()+1;k++){
					if (k < 0 || k > this.numPtosY-1){//comprobamos que no sobrepasamos los límites de la rejilla
//						System.out.println("Desborde de la rejilla en el eje y, k vale "+k);
						continue;
					}
					if (this.rejilla[j][k].isClosedSet()){
//						System.out.println("está en el closedSet");
						continue; // si el nodo está en el closedSet no hacemos nada con el y seguimos mirando						
					}
					//Clonamos el vector de obstáculos
					this.setObstaculos(this.sim.getObstaculos());
					//Calculamos el tiempo que el vehículo va a alcanzar el obstáculo
					double t = actual.getG_score()/this.getVelCoche();
//					System.out.println("tiempo t "+t);
					//Calculamos la posición de los obstáculos un tiempo t después
//					setObstaculos(this.getSim().moverObstaculos(t,this.getObstaculos()));
					this.getSim().moverObstaculos(t,this.getObstaculos());
					//limpiamos las celdas anteriores marcadas con obstáculos
					clearObstacles();
					//Los marcamos en la rejilla
					addObstacles(this.getObstaculos());
					if (this.rejilla[j][k].isOccupied()){
//						System.out.println("la celda está ocupada por un obstáculo");
						continue; // si el nodo está en el ocupado por un obstáculo no hacemos nada con el y seguimos mirando						
					}
					double g_score_tentativo = actual.getG_score() +
							actual.distThisPoint2Point(this.rejilla[j][k].getxPosition(),this.rejilla[j][k].getyPosition());
					//Comprobamos si el vecino está en el openSet
					if (!this.rejilla[j][k].isOpenSet()){
//						System.out.println("el vecino no está en el openset y lo deberíamos meter");
						//si no está lo metemos en el openSet
						this.rejilla[j][k].setOpenSet(true);
						this.rejilla[j][k].setClosedSet(false);
						openSet.add(this.rejilla[j][k]);
						//Calculamos su h_score (distancia euclídea hasta el objetivo) y se lo asignamos
						this.rejilla[j][k].setH_score(this.rejilla[j][k].distThisPoint2Point(goal));
						tentative_is_better = true;
					}else if (g_score_tentativo < this.rejilla[j][k].getG_score()){
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
						this.rejilla[j][k].setCameFrom(actual);
						this.rejilla[j][k].setG_score(g_score_tentativo);
						this.rejilla[j][k].setF_score(this.rejilla[j][k].getG_score()+this.rejilla[j][k].getH_score());						
					}
				}
			}
		}
		if (!caminoCompleto){
			camino = reconstruirCaminoAEstrella(actual);
			System.out.println("no se logró un camino completo con grid");
		}
		return camino;
	}
	
public Vector<Matrix> busquedaAEstrellaConMarcado(double RadioBusqueda){
		
		boolean caminoCompleto = false;
		boolean tentative_is_better = false;		
		double minF_score = Double.POSITIVE_INFINITY;
		int indMin = 0;
		Vector<Matrix> camino = new Vector<Matrix>();
		Vector<GridSearchPoint> openSet = new Vector<GridSearchPoint>();
		Vector<GridSearchPoint> closedSet = new Vector<GridSearchPoint>();		
		openSet.clear();
		closedSet.clear();	
		clearSearchData();//Desmarcamos las celdas marcadas en la iteración anterior como pertenecientes al open o closedset
		int [] start = getStartPos();
//		System.out.println("la posición del coche es (en búsqueda A*) "+start[0]+" "+start[1]);
		int [] goal = getGoalPos();	
		this.rejilla[start[0]][start[1]].setG_score(0);
		this.rejilla[start[0]][start[1]].setH_score(this.rejilla[start[0]][start[1]].distThisPoint2Point(goal[0],goal[1]));
		this.rejilla[start[0]][start[1]].setF_score(this.rejilla[start[0]][start[1]].getG_score()+
				this.rejilla[start[0]][start[1]].getH_score());
		this.rejilla[start[0]][start[1]].setOpenSet(true);
		openSet.add(this.rejilla[start[0]][start[1]]);
		GridSearchPoint actual = this.rejilla[start[0]][start[1]];		
		
		//--------MARCO LA REJILLA CON LA INFORMACIÓN DE LOS BOIDS------------------------		
		
		//limpiamos las celdas anteriores marcadas con obstáculos
//		clearObstacles();
		//Clonamos el vector de boids
		this.setBoids(this.sim.getBandada());
		//Marcamos todas las celdas como ocupadas
		marcarTodoOcupado();
		//Marcamos las celdas próximas a los boids como libres y les asignamos un peso
		addBoids(this.getBoids(), RadioBusqueda);
		//Marcamos como ocupadas las celdas que intersectan con la superficie de los obstáculos
		addObstacles(this.getObstaculos());
		
		
		
		while (!openSet.isEmpty()){
//			System.out.println("tamaño del openset "+openSet.size());
//			System.out.println("tamaño del closedSet "+closedSet.size());
			minF_score = Double.POSITIVE_INFINITY;
			//Buscamos el nodo del opneSet con mejor f_score
			for (int i=0;i<openSet.size();i++){
				if(openSet.elementAt(i).getF_score() < minF_score) {
					indMin=i;
					minF_score=openSet.elementAt(i).getF_score();
				}								
			}
			actual = openSet.elementAt(indMin);
//			System.out.println("índices del punto actual "+actual.getxIndex()+" "+actual.getyIndex()+" y su f_score "+actual.getF_score());
			//comprobamos si el nodo actual es el objetivo
			if (Math.abs(actual.getxIndex() - goal[0]) < 2 && Math.abs(actual.getyIndex() - goal[1]) < 2){ //Asegurarse de que la comprobación funciona
//			if (actual.getxIndex() == goal[0] && actual.getyIndex() == goal[1]){ //Asegurarse de que la comprobación funciona
				caminoCompleto = true;
//				System.out.println("se completó con éxito el camino, supuestamente");
				return reconstruirCaminoAEstrella(actual);
				
			}
			//Quitamos el nodo actual del openSet 
			openSet.remove(indMin); //También existe un método para quitar un elemento especificando que objeto hay que quitar
			//y lo añadimos al closedSet
			actual.setOpenSet(false);
			actual.setClosedSet(true);
			closedSet.add(actual);
//			System.out.println("tamaño de la rejilla "+this.getNumPtosX()+" "+this.getNumPtosY());
			//Exploramos los vecinos del actual
			for (int j=actual.getxIndex()-1;j<=actual.getxIndex()+1;j++){
//				System.out.println("posición de goal "+goal[0]+" "+goal[1]);
				if (j < 0 || j > this.numPtosX-1){//comprobamos que no sobrepasamos los límites de la rejilla
//					System.out.println("Desborde de la rejilla en el eje x, j vale "+j);
					continue;
				}				
				for (int k=actual.getyIndex()-1;k<=actual.getyIndex()+1;k++){
					if (k < 0 || k > this.numPtosY-1){//comprobamos que no sobrepasamos los límites de la rejilla
//						System.out.println("Desborde de la rejilla en el eje y, k vale "+k);
						continue;
					}
					if (this.rejilla[j][k].isClosedSet()){
//						System.out.println("está en el closedSet");
						continue; // si el nodo está en el closedSet no hacemos nada con el y seguimos mirando						
					}
					//Clonamos el vector de obstáculos
//					this.setObstaculos(this.sim.getObstaculos());
					
					
					//Calculamos el tiempo que el vehículo va a alcanzar el obstáculo
//					double t = actual.getG_score()/this.getVelCoche();
////					System.out.println("tiempo t "+t);
//					//Calculamos la posición de los obstáculos un tiempo t después
//					setObstaculos(this.getSim().moverObstaculos(t,this.getObstaculos()));
//					this.getSim().moverObstaculos(t,this.getObstaculos());
					
					
					//limpiamos las celdas anteriores marcadas con obstáculos
//					clearObstacles();
//					//Clonamos el vector de boids
//					this.setBoids(this.sim.getBandada());
//					//Marcamos todas las celdas como ocupadas
//					marcarTodoOcupado();
//					//Marcamos las celdas próximas a los boids como libres y les asignamos un peso
//					addBoids(this.getBoids(), RadioBusqueda);
					//Marcamos como ocupadas las celdas que intersectan con la superficie de los obstáculos
//					addObstacles(this.getObstaculos());
					if (this.rejilla[j][k].isOccupied()){
//						System.out.println("la celda está ocupada por un obstáculo");
						continue; // si el nodo está en el ocupado por un obstáculo no hacemos nada con el y seguimos mirando						
					}
					double g_score_tentativo = actual.getG_score() +
							actual.distThisPoint2Point(this.rejilla[j][k].getxPosition(),this.rejilla[j][k].getyPosition());
					//Comprobamos si el vecino está en el openSet
					if (!this.rejilla[j][k].isOpenSet()){
//						System.out.println("el vecino no está en el openset y lo deberíamos meter");
						//si no está lo metemos en el openSet
						this.rejilla[j][k].setOpenSet(true);
						this.rejilla[j][k].setClosedSet(false);
						openSet.add(this.rejilla[j][k]);
						//Calculamos su h_score (distancia euclídea hasta el objetivo) y se lo asignamos
						this.rejilla[j][k].setH_score(this.rejilla[j][k].distThisPoint2Point(goal));
						tentative_is_better = true;
					}else if (g_score_tentativo < this.rejilla[j][k].getG_score()){
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
						this.rejilla[j][k].setCameFrom(actual);
						this.rejilla[j][k].setG_score(g_score_tentativo);
						this.rejilla[j][k].setF_score(this.rejilla[j][k].getG_score()+this.rejilla[j][k].getH_score()+this.rejilla[j][k].getPeso_marcado());						
//						this.rejilla[j][k].setF_score(this.rejilla[j][k].getG_score()+this.rejilla[j][k].getH_score());
					}
				}
			}
		}
		if (!caminoCompleto){
			System.out.println("no se logró un camino completo con grid");
			camino = reconstruirCaminoAEstrella(actual);			
		}
		return camino;
	}
	
	public Vector<Matrix> reconstruirCaminoAEstrella(GridSearchPoint actual){
		Vector<Matrix> camino = new Vector<Matrix>();
		GridSearchPoint aux = actual;
		while(aux.getCameFrom() != null){
			double pos[] = {aux.getxPosition(),aux.getyPosition()};
			Matrix posi = new Matrix(pos,2);
			camino.add(posi);
			aux = aux.getCameFrom();			
		}
//		System.out.println("tamaño del camino a estrella "+ camino.size());
//		double pos[] = {(double)this.startPos[0],(double)this.startPos[1]};
//		Matrix posi = new Matrix(pos,2);
//		camino.add(posi);
		return camino;
	}

	public void clearSearchData(){
		for (int i=0;i<getNumPtosX();i++){
			for (int j=0;j<getNumPtosY();j++){
				this.rejilla[i][j].setClosedSet(false);
				this.rejilla[i][j].setOpenSet(false);
			}
		}
	}
	
	public void clearObstacles(){
		for (int i=0;i<getNumPtosX();i++){
			for (int j=0;j<getNumPtosY();j++){
				this.rejilla[i][j].setOccupied(false);
			}
		}
	}
	/**
	 * Se marcan todas las celdas como ocupadas y se les asigna un valor de peso de marcado infinito
	 * Cuando se añadan los boids a la rejilla, las celdas próximas a cad aboid se marcarán como libres y
	 * se les asignará un valor de peso de marcado igual a la distancia a la que se encuentren del boid más cercano
	 */
	public void marcarTodoOcupado(){
		for (int i=0;i<getNumPtosX();i++){
			for (int j=0;j<getNumPtosY();j++){
				this.rejilla[i][j].setOccupied(true);
				this.rejilla[i][j].setPeso_marcado(Double.POSITIVE_INFINITY);				
			}
		}
	}
	
	public void clearSearchDataAndObst(){
		for (int i=0;i<getNumPtosX();i++){
			for (int j=0;j<getNumPtosY();j++){
				this.rejilla[i][j].setClosedSet(false);
				this.rejilla[i][j].setOpenSet(false);
				this.rejilla[i][j].setOccupied(false);
			}
		}
	}
	
	public void clearVectorParticulas(){
		for (int i=0;i<getNumPtosX();i++){
			for (int j=0;j<getNumPtosY();j++){
				this.rejilla[i][j].particulas.clear();
			}
		}
	}
	
	
	public void clearVectorObstaculos(){
		for (int i=0;i<getNumPtosX();i++){
			for (int j=0;j<getNumPtosY();j++){	
				if(!this.rejilla[i][j].obstaculos.isEmpty()){
					this.rejilla[i][j].obstaculos.clear();
				}				
			}
		}
	}
	
	public double getVelCoche() {
		return velCoche;
	}

	public void setVelCoche(double velCoche) {
		this.velCoche = velCoche;
	}
	
	public Vector<Obstaculo> getObstaculos() {
		return obstaculos;
	}
	public Vector<Boid> getBoids() {
		return boids;
	}
	/**
	 * Copia el vector de obstáculos
	 * @param obstaculos Vector que contiene los obstáculos presentes en el escenario
	 */
	public void setObstaculos(Vector<Obstaculo> obstaculos) {
		if (!this.obstaculos.isEmpty()){
			this.obstaculos.clear();
		}
		
		for (int i=0;i<obstaculos.size();i++){
			double posX = obstaculos.elementAt(i).getPosicion().get(0, 0);
			double posY = obstaculos.elementAt(i).getPosicion().get(1, 0);
			double velX = obstaculos.elementAt(i).getVelocidad().get(0, 0);
			double velY = obstaculos.elementAt(i).getVelocidad().get(1, 0);
			double vecRumboX = obstaculos.elementAt(i).getRumboDeseado().get(0, 0);
			double vecRumboY = obstaculos.elementAt(i).getRumboDeseado().get(1, 0);
					
			Obstaculo obs = new Obstaculo(posX, posY, velX, velY, vecRumboX, vecRumboY);
			this.obstaculos.add(obs);
		}
	}

	/**
	 * Copia el vector de boids
	 * @param bandada
	 */
	public void setBoids(Vector<Boid> bandada) {
		if (!this.boids.isEmpty()){			
			this.boids.clear();
		}
		
		for (int i=0;i<bandada.size();i++){
			double posX = bandada.elementAt(i).getPosicion().get(0, 0);
			double posY = bandada.elementAt(i).getPosicion().get(1, 0);
			double velX = bandada.elementAt(i).getVelocidad().get(0, 0);
			double velY = bandada.elementAt(i).getVelocidad().get(1, 0);
			double acelX = bandada.elementAt(i).getAceleracion().get(0, 0);
			double acelY = bandada.elementAt(i).getAceleracion().get(1, 0);
			double posi[] = {posX,posY};
			Matrix pos = new Matrix(posi,2);
			double velo[] = {velX,velY};
			Matrix vel = new Matrix(velo,2);
			double acele[] = {acelX,acelY};
			Matrix acel = new Matrix(acele,2);
			Boid boid = new Boid(pos,vel,acel);
			
			this.boids.add(boid);
		}
	}
	
	public Simulador getSim() {
		return sim;
	}

	public void setSim(Simulador sim) {
		this.sim = sim;
	}
	
	public double getResolution() {
		return resolution;
	}

	public void setResolution(double resolution) {
		this.resolution = resolution;
	}
	
	public double getLongitudX() {
		return longitudX;
	}

	public void setLongitudX(double longitudX) {
		this.longitudX = longitudX;
	}

	public double getLongitudY() {
		return longitudY;
	}

	public void setLongitudY(double longitudY) {
		this.longitudY = longitudY;
	}

	public int getNumPtosX() {
		return numPtosX;
	}

	public void setNumPtosX(int numPtosX) {
		this.numPtosX = numPtosX;
	}

	public int getNumPtosY() {
		return numPtosY;
	}

	public void setNumPtosY(int numPtosY) {
		this.numPtosY = numPtosY;
	}

	public GridSearchPoint[][] getRejilla() {
		return rejilla;
	}

	public void setRejilla(GridSearchPoint[][] rejilla) {
		this.rejilla = rejilla;
	}
	
	public int[] getGoalPos() {
		return goalPos;
	}

	public void setGoalPos(int[] goalPos) {
		this.goalPos = goalPos;
	}
	
	public void setGoalPos(double posX,double posY) {
		int[] goalPosi = new int[2];
		goalPosi[0] = (int)Math.floor(posX/getResolution());
		goalPosi[1] = (int)Math.floor(posY/getResolution());
		setGoalPos(goalPosi);
	}

	public int[] getStartPos() {
		return startPos;
	}

	public void setStartPos(int[] startPos) {
		this.startPos = startPos;
	}
	
	public void setStartPos(double posX,double posY) {
		int[] startPosi = new int[2];
		startPosi[0] = (int)Math.floor(posX/getResolution());
		startPosi[1] = (int)Math.floor(posY/getResolution());
//		System.out.println("posición de start "+startPosi[0]+" "+startPosi[1]);
		setStartPos(startPosi);
	}
	


}
