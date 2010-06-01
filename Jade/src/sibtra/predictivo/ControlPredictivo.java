/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package sibtra.predictivo;

import sibtra.gps.Trayectoria;
import sibtra.log.LoggerArrayDoubles;
import sibtra.log.LoggerFactory;
import sibtra.util.UtilCalculos;
import Jama.Matrix;

/**
 * Esta clase implementa el algoritmo de control predictivo DMC
 * @author Jesus
 */
public class ControlPredictivo {
    private static final double minAvancePrediccion = 2;
	/**
     * Variable que indica hasta donde llegará la predicción
     */
    int horPrediccion;
    /**
     * Indica cuantos comandos de control futuros se quieren calcular
     */
    int horControl;
 
    /** trayectoria que debe seguir el coche */
    Trayectoria ruta=null;
    
    /**
     * Dependiendo de su valor se le da más importancia a minimizar el error o 
     * a minimizar el gasto de comando. Si es igual a 0 no tiene en cuenta el gasto de 
     * comando, por lo que el controlador probablemente genere comandos bruscos.
     * En cambio si le damos un valor muy alto, restringimos mucho el uso de comando, 
     * con lo que se produce una actuación muy suave y lenta del controlador. Hay que 
     * alcanzar un término medio que permita actuar suficientemente rápido pero sin 
     * sobrepasamientos
     */    
    double landa;
    double pesoError = 0.8;
    Coche carroOriginal=null;
    /**
     * Objeto que se copiará del carro original más actualizado en cada llamada
     * a {@link #calculaPrediccion(double, double) }
     */
    Coche carroSim=null;
    /**
     * Periodo de muestreo del controlador
     */
    double Ts;
    Matrix G;
    private Matrix landaEye;
    
    /** Almacena las orientaciones de la prediccion*/
    double[] predicOrientacion;
    /** Almacena las posiciones x e y de la prediccion*/
    double[][] prediccionPosPorFilas;
    /** Último comando calculado por el controlador predictivo*/
    double comandoCalculado;
    /** Distancia lateral mínima del coche a la ruta */ 
	double distanciaLateral;
	
	/** Se usa en calculos internos pero la tenemos como campo para no pedir memoria en cada iteración */
	private double[] vectorError;
	/** Se usa en calculos internos pero la tenemos como campo para no pedir memoria en cada iteración */
	private double[] orientacionesDeseadas;
	/** Se usa en calculos internos pero la tenemos como campo para no pedir memoria en cada iteración */
	private double[] respuestaEscalon;
	private Coche carroEscalon;
	/** Inidica el índice del punto de la ruta más cercano al coche de la iteración anterior*/
	int indMinAnt;
	/** Sirve para dar más peso a los componentes más cercanos al instante actual del 
	 * vector de errores futuros o viceversa.
	 * si alpha es >1 se pesan más los coeficientes más próximos al instante actual
     * si alpha está entre 0 y 1 se pesan más los instantes más alejados*/
	private double alpha = 1.05;
	/** Hace las veces de ganancia proporcional al error
	 * 
	 */
	private LoggerArrayDoubles logPredicOrientacion;
	private LoggerArrayDoubles logPredicPosX;
	private LoggerArrayDoubles logPredicPosY;
	private LoggerArrayDoubles logVectorError;
	private LoggerArrayDoubles logParametros;

	/**
     * 
     * @param carroOri Puntero al modelo actualzado del vehículo. No se modificará 
     * en esta clase
     * @param horPrediccion Horizonte de predicción
     * @param horControl Horizonte de control
     * @param landa Peso 
     */
    public ControlPredictivo(Coche carroOri,Trayectoria ruta,int horPrediccion,int horControl,double landa,double Ts){
        carroOriginal = carroOri;
        carroSim = (Coche)carroOri.clone();
        carroEscalon=(Coche)carroOriginal.clone();
        this.horPrediccion = horPrediccion;
        this.horControl = horControl;
        this.landa = landa;
        this.ruta = ruta;
        this.Ts = Ts;
        this.indMinAnt = -1;
        
        // creamos todas las matrices que se usan en las iteracioner para evitar tener que 
        //pedir memoria cada vez
        G = new Matrix(horPrediccion,horControl);
        this.landaEye = Matrix.identity(horControl,horControl).times(landa);
        prediccionPosPorFilas = new double[2][horPrediccion];
        predicOrientacion = new double[horPrediccion];
        vectorError = new double[horPrediccion];
        orientacionesDeseadas = new double[horPrediccion];
        respuestaEscalon = new double[horPrediccion];
        
        logPredicOrientacion=LoggerFactory.nuevoLoggerArrayDoubles(this, "PrediccionOrientacion");
        logPredicPosX=LoggerFactory.nuevoLoggerArrayDoubles(this, "PrediccionPosicionX");
        logPredicPosY=LoggerFactory.nuevoLoggerArrayDoubles(this, "PrediccionPosicionY");
        logVectorError=LoggerFactory.nuevoLoggerArrayDoubles(this, "VectorError");
        logParametros=LoggerFactory.nuevoLoggerArrayDoubles(this, "ParametrosPredictivo");
        logParametros.add(horControl,horPrediccion,landa,alpha,pesoError);
        logParametros.setDescripcion("[horControl,horPrediccion,landa,alpha,pesoError]");

    }

    public double getPesoError() {		
		return pesoError;
	}
	public void setAlpha(double alpha2) {
		alpha = alpha2;		
        logParametros.add(horControl,horPrediccion,landa,alpha,pesoError);
	}
	public void setPesoError(double pesoError2) {
		pesoError = pesoError2;		
        logParametros.add(horControl,horPrediccion,landa,alpha,pesoError);
	}
    /** @return Devuelve el primer componente del vector {@link #orientacionesDeseadas} */
    public double getOrientacionDeseada() {
		return orientacionesDeseadas[0];
	}

	public int getHorControl() {
		return horControl;
	}
	
	/** establece horizonte de control, recalculando {@link #G} si es necesario */
	public void setHorControl(int horControl) {
        logParametros.add(horControl,horPrediccion,landa,alpha,pesoError);
		if (horControl==this.horControl)
			return;
		this.horControl = horControl;
        G = new Matrix(horPrediccion,horControl);
        this.landaEye = Matrix.identity(horControl,horControl).times(landa);
	}
	
	public int getHorPrediccion() {
		return horPrediccion;
	}
	
	/** establece horizonte de predicción, recalculando {@link #G} s es necesario */
	public void setHorPrediccion(int horPrediccion) {
        logParametros.add(horControl,horPrediccion,landa,alpha,pesoError);
		if(horPrediccion==this.horPrediccion)
			return;
		this.horPrediccion = horPrediccion;
        prediccionPosPorFilas = new double[2][horPrediccion];
        G = new Matrix(horPrediccion,horControl);
        predicOrientacion = new double[horPrediccion];
        vectorError = new double[horPrediccion];
        orientacionesDeseadas = new double[horPrediccion];
        respuestaEscalon = new double[horPrediccion];
	}
	
	public double getLanda() {
		return landa;
	}
	
	public void setLanda(double landa) {
        logParametros.add(horControl,horPrediccion,landa);
		if(landa==this.landa)
			return;
		this.landa = landa;
        this.landaEye = Matrix.identity(horControl,horControl).times(landa);
	}
	
	public void setRuta(Trayectoria nuevaRuta){
		this.indMinAnt=-1;
		this.ruta = nuevaRuta;
	}

    /** Se encargará de inicializar todas las variables del control predictivo que 
     * tengan que tener un valor concreto al inicio de una navegación. Tener en cuenta
     * que se entiende como un inicio de navegación cada vez que el usuario pulse 
     * en el checkBox navegando. Esto permite conducir manualmente el vehículo hasta
     * otro punto después de haber realizado una navegación automática y en este nuevo
     * punto volver a conectar el modo automático
     * */
    public void iniciaNavega() {
    	// Al poner a -1 este índice se le indica al método calculaDistMinOptimizado
    	// que haga una búsqueda exaustiva por todos los puntos de la ruta
    	indMinAnt = -1;
    		
    }
    
    private void calculaHorizonte(){    	
    	double metrosAvanzados = carroOriginal.getVelocidad()*Ts*horPrediccion;
    	double velMin = 0.1;
    	if (metrosAvanzados <= minAvancePrediccion){
    		//calculo que horizonte de predicción es necesario para que la 
    		//predicción avance como mínimo minAvancePrediccion metros
    		if (carroOriginal.getVelocidad() > 0){
    			horPrediccion = (int)Math.ceil(minAvancePrediccion/(carroOriginal.getVelocidad()*Ts));
    		}else{    		
    			if (carroOriginal.getVelocidad() == 0){
    				//TODO ver que hacemos con el caso de que la 
    				// velocidad sea negativa o igual a cero
    				//Evitamos la divisón por cero
    				//horPrediccion = (int)Math.ceil(minAvancePrediccion/(velMin*Ts));
    			}
    			else{
    				//caso en que la velocidad del coche es negativa
    			}
    		}
    	}
    }
    
    /**
     * Calcula la evolución del modelo del vehículo tantos pasos hacia delante como
     * horizonte de predicción se haya definido
     * @param comando Consigna para la orientación del volante
     * @param velocidad Velocidad lineal del vehículo
     * @return Vector de errores futuros entre las orientaciones alcanzas por 
     * el vehículo en la predicción y el vectores de orientaciones deseadas futuras
     */
    private double[] calculaPrediccion(double comando,double velocidad){
        carroSim.copy(carroOriginal);//
        /*Hacemos la copia del objeto carroOriginal para realizar la simulación
         sobre la copia, de manera que no se modifiquen los datos reales*/
        //La siguiente linea de código es el cálculo del vector deseado en MAtlab
//        vec_deseado(1,:) = k_dist*(pos_ref(mod(ind_min(k)+cerca,length(pos_ref))+1,:) - [carro.x,carro.y])
//+ k_ang*[cos(tita_ref(mod(ind_min(k)+cerca,length(tita_ref))+1)),sin(tita_ref(mod(ind_min(k)+cerca,length(tita_ref))+1))];
//        int indMin = calculaDistMin(ruta,carroSim.getX(),carroSim.getY());       
        //TODO usar directamente indiceMasCercanoOptimizado ya que es la posicion actual del coche
        
        indMinAnt = ruta.indiceMasCercanoOptimizado(carroSim.getX(),carroSim.getY(),indMinAnt);
        
//        ruta.situaCoche(carroSim.getX(),carroSim.getY());
//        indMinAnt = ruta.indiceMasCercano();
        double dx=ruta.x[indMinAnt]-carroSim.getX();
        double dy=ruta.y[indMinAnt]-carroSim.getY();
        distanciaLateral=Math.sqrt(dx*dx+dy*dy);
        double vectorDeseadoX = ruta.x[indMinAnt] - carroSim.getX() + Math.cos(ruta.rumbo[indMinAnt]);
        double vectorDeseadoY = ruta.y[indMinAnt] - carroSim.getY() + Math.sin(ruta.rumbo[indMinAnt]);
        orientacionesDeseadas[0] = Math.atan2(vectorDeseadoX,vectorDeseadoY);
        predicOrientacion[0] = carroSim.getYaw();
        vectorError[0] = orientacionesDeseadas[0] - predicOrientacion[0];
        prediccionPosPorFilas[0][0] = carroSim.getX();
        prediccionPosPorFilas[1][0] = carroSim.getY();
        int indMin = indMinAnt;
        for (int i=1; i<horPrediccion;i++ ){
            carroSim.calculaEvolucion(comando,velocidad,Ts);
            predicOrientacion[i] = carroSim.getYaw();
            indMin = ruta.indiceMasCercanoOptimizado(carroSim.getX(),carroSim.getY(),indMin);
//            indMin = calculaDistMin(ruta,carroSim.getX(),carroSim.getY());
            vectorDeseadoX = ruta.x[indMin] - carroSim.getX() + Math.cos(ruta.rumbo[indMin]);
            vectorDeseadoY = ruta.y[indMin] - carroSim.getY() + Math.sin(ruta.rumbo[indMin]);
            orientacionesDeseadas[i] = Math.atan2(vectorDeseadoY,vectorDeseadoX);
//            System.out.println("Indice minimo " + indMin);
//            System.out.println("Vector x "+vectorDeseadoX+" "+ "Vector y "+vectorDeseadoY+"\n");
//            System.out.println("Orientacion deseada " + orientacionesDeseadas[i] + " " 
//                                +"prediccion de orientacion " + predicOrientacion[i]+"\n");
            // coefError pesa los valores del vectorError dependiendo del valor de alpha
            // si alpha es >1 se pesan más los coeficientes más próximos al instante actual
            // si alpha está entre 0 y 1 se pesan más los instantes más alejados
            double coefError = Math.pow(pesoError*alpha,horPrediccion-i);
            vectorError[i] = coefError*(UtilCalculos.normalizaAngulo(orientacionesDeseadas[i] - predicOrientacion[i]));
            prediccionPosPorFilas[0][i] = carroSim.getX();
            prediccionPosPorFilas[1][i] = carroSim.getY();
        }
        
        logPredicOrientacion.add(predicOrientacion);
        logPredicPosX.add(prediccionPosPorFilas[0]);
        logPredicPosY.add(prediccionPosPorFilas[1]);
        logVectorError.add(vectorError);
        return vectorError;
    }
    
    
    public double calculaComando(){
        //    vector_error = tita_deseado - ftita;
//    vector_error = vector_error + (vector_error > pi)*(-2*pi) + (vector_error < -pi)*2*pi;
        double[] vectorError = calculaPrediccion(carroOriginal.getConsignaVolante()
                                                   ,carroOriginal.getVelocidad());
        //    M = diag(peso_tita'*vector_error);
        Matrix M = new Matrix(vectorError,horPrediccion).transpose();
        //M.print(1,6);
//        u = inv(G'*G + landa*eye(hor_control))*G'*M;
//    u_actual = u(1) + u_anterior;
//        %-----------Cálculo del vector de errores----------
        calculaG();
        //G.print(1,6);
        Matrix Gt = G.transpose();
//        Matrix GtporM = Gt.times(M.transpose());
//        Matrix GtporG = Gt.times(G);
//        Matrix masLandaEye = GtporG.plus(landaEye);
        Matrix vectorU = Gt.times(G).plus(landaEye).inverse().times(Gt).times(M.transpose());
        //vectorU.print(1,6);
        comandoCalculado = vectorU.get(0,0) +  carroOriginal.getConsignaVolante();
        return comandoCalculado;
       
    }    
    
    private Matrix calculaG(){
        //TODO Optimizar el cálculo de G para que solo se realice si cambia la velocidad
        // El cambio en la velocidad se acotará a unos niveles mínimos
        double[] escalon = calculaEscalon(carroOriginal.getVelocidad());
//        for j=1:hor_control,
//    cont=1;
//    for i=j:hor_predic
//        Gtita(i,j)=escalon_tita(cont);
//        cont=cont+1;
//    end
//end
        for (int j=0;j<horControl;j++){
            int cont = 0;
            for (int i=j;i<horPrediccion;i++){
            G.set(i,j,escalon[cont]);
            cont++;
            }
        }
        return G;
    }
    
    private double[] calculaEscalon(double velocidad){
//    	carroEscalon.copy(carroOriginal);
        carroEscalon.setPostura(0,0,0,0);
        carroEscalon.setEstadoA0();
        
        for (int i=0;i<horPrediccion;i++){
            carroEscalon.calculaEvolucion(Math.PI/6,velocidad, Ts);
            respuestaEscalon[i] = carroEscalon.getYaw();
        }
        return respuestaEscalon;
    }
    
    protected void finalize() throws Throwable {
    	//borramos todos los loggers
    	LoggerFactory.borraLogger(logPredicOrientacion);
    	LoggerFactory.borraLogger(logPredicPosX);
    	LoggerFactory.borraLogger(logPredicPosY);
    	LoggerFactory.borraLogger(logVectorError);
    	LoggerFactory.borraLogger(logParametros);
    	super.finalize();
    }
		
}
