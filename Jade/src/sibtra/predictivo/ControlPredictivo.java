/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package sibtra.predictivo;

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
    /**
     * Array de doubles bidimensional con tantas filas como puntos
     * tenga la trayectoria y tres columnas, la primera para la coordenada
     * x, la segunda para la coordenada y, la tercera para la orientación del
     * coche en cada punto
     */
    double[][] ruta;
    /**
     * Indica si la ruta está cerrada o no
     */
    private boolean rutaCerrada;
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
    Coche carroOriginal;
    /**
     * Objeto que se copiará del carro original más actualizado en cada llamada
     * a {@link #calculaPrediccion(double, double, double) }
     */
    Coche carroSim;
    /**
     * Periodo de muestreo del controlador
     */
    double Ts;
    Matrix G;
    private Matrix landaEye;
    
    /** Almacena las orientaciones de la prediccion*/
    double[] predicOrientacion;
    /** Almacena las posiciones x e y de la prediccion*/
    double[][] prediccionPosicion;
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
	private int indMinAnt;
	/** Sirve para dar más peso a los componentes más cercanos al instante actual del 
	 * vector de errores futuros o viceversa.
	 * si alpha es >1 se pesan más los coeficientes más próximos al instante actual
     * si alpha está entre 0 y 1 se pesan más los instantes más alejados*/
	private double alpha = 1.05;
	/** Hace las veces de ganancia proporcional al error
	 * 
	 */

	/**
     * 
     * @param carroOri Puntero al modelo actualzado del vehículo. No se modificará 
     * en esta clase
     * @param horPred Horizonte de predicción
     * @param horCont Horizonte de control
     * @param landa Peso 
     */
    public ControlPredictivo(Coche carroOri,double[][] ruta,int horPrediccion,int horControl,double landa,double Ts){
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
        prediccionPosicion = new double[horPrediccion][2];
        predicOrientacion = new double[horPrediccion];
        vectorError = new double[horPrediccion];
        orientacionesDeseadas = new double[horPrediccion];
        respuestaEscalon = new double[horPrediccion];
/**
 * Constructor donde también se le pasa la información de si la ruta está cerrada o no
 */
    }
    public ControlPredictivo(Coche carroOri,double[][] ruta,int horPrediccion,int horControl,double landa,double Ts,boolean rutaCerrada){
        carroOriginal = carroOri;
        carroSim = (Coche)carroOri.clone();
        carroEscalon=(Coche)carroOriginal.clone();
        this.horPrediccion = horPrediccion;
        this.horControl = horControl;
        this.landa = landa;
        this.ruta = ruta;
        this.Ts = Ts;
        this.rutaCerrada = rutaCerrada;
        this.indMinAnt = -1;
        
        
        // creamos todas las matrices que se usan en las iteraciones para evitar tener que 
        //pedir memoria cada vez
        G = new Matrix(horPrediccion,horControl);
        this.landaEye = Matrix.identity(horControl,horControl).times(landa);
        prediccionPosicion = new double[horPrediccion][2];
        predicOrientacion = new double[horPrediccion];
        vectorError = new double[horPrediccion];
        orientacionesDeseadas = new double[horPrediccion];
        respuestaEscalon = new double[horPrediccion];

    }
	public double getPesoError() {		
		return pesoError;
	}
	public void setAlpha(double alpha2) {
		alpha = alpha2;		
	}
	public void setPesoError(double pesoError2) {
		pesoError = pesoError2;		
	}
    /**
     * 
     * @return Devuelve el primer componente del vector {@link #orientacionesDeseadas}
     */
    public double getOrientacionDeseada() {
		return orientacionesDeseadas[0];
	}

	public int getHorControl() {
		return horControl;
	}
	public void setHorControl(int horControl) {
		if (horControl==this.horControl)
			return;
		this.horControl = horControl;
        G = new Matrix(horPrediccion,horControl);
        this.landaEye = Matrix.identity(horControl,horControl).times(landa);
	}
	public int getHorPrediccion() {
		return horPrediccion;
	}
	public void setHorPrediccion(int horPrediccion) {
		if(horPrediccion==this.horPrediccion)
			return;
		this.horPrediccion = horPrediccion;
        prediccionPosicion = new double[horPrediccion][2];
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
		if(landa==this.landa)
			return;
		this.landa = landa;
        this.landaEye = Matrix.identity(horControl,horControl).times(landa);
	}
        public void setRuta(double[][] nuevaRuta){
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
     * @param carroSim Clase que modela el vehículo
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
        indMinAnt = UtilCalculos.indiceMasCercanoOptimizado(ruta, rutaCerrada, carroSim.getX(),carroSim.getY(),indMinAnt);
        double dx=ruta[indMinAnt][0]-carroSim.getX();
        double dy=ruta[indMinAnt][1]-carroSim.getY();
        distanciaLateral=Math.sqrt(dx*dx+dy*dy);
        double vectorDeseadoX = ruta[indMinAnt][0] - carroSim.getX() + Math.cos(ruta[indMinAnt][2]);
        double vectorDeseadoY = ruta[indMinAnt][1] - carroSim.getY() + Math.sin(ruta[indMinAnt][2]);
        orientacionesDeseadas[0] = Math.atan2(vectorDeseadoX,vectorDeseadoY);
        predicOrientacion[0] = carroSim.getTita();
        vectorError[0] = orientacionesDeseadas[0] - predicOrientacion[0];
        prediccionPosicion[0][0] = carroSim.getX();
        prediccionPosicion[0][1] = carroSim.getY();
        int indMin = indMinAnt;
        for (int i=1; i<horPrediccion;i++ ){
            carroSim.calculaEvolucion(comando,velocidad,Ts);
            predicOrientacion[i] = carroSim.getTita();
            indMin = UtilCalculos.indiceMasCercanoOptimizado(ruta, rutaCerrada, carroSim.getX(),carroSim.getY(),indMin);
//            indMin = calculaDistMin(ruta,carroSim.getX(),carroSim.getY());
            vectorDeseadoX = ruta[indMin][0] - carroSim.getX() + Math.cos(ruta[indMin][2]);
            vectorDeseadoY = ruta[indMin][1] - carroSim.getY() + Math.sin(ruta[indMin][2]);
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
            prediccionPosicion[i][0] = carroSim.getX();
            prediccionPosicion[i][1] = carroSim.getY();
        }
        
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
            respuestaEscalon[i] = carroEscalon.getTita();
        }
        return respuestaEscalon;
    }
    static double[][] generaRuta(int numPuntos,double intervalo){
        double[][] rutaAux = new double[numPuntos][3];
        rutaAux[0][0] = 0;
        rutaAux[0][1] = 0;
        rutaAux[0][2] = 0;
        for (int i=1;i<numPuntos;i++){
            rutaAux[i][0] = i*intervalo;
            rutaAux[i][1] = rutaAux[i][0]*2;
            rutaAux[i][2] = Math.atan2((rutaAux[i][1]-rutaAux[i-1][1]),
                                        (rutaAux[i][0]-rutaAux[i-1][0]));
        }
        return rutaAux;
    }
    public static void main(String[] args){
        Coche carroOri = new Coche();
        double vel = 2;
        double consVolante = 0;
        carroOri.setVelocidad(vel);
        carroOri.setConsignaVolante(consVolante);
        int horPredic = 12;
        int horCont = 3;
        double paramLanda = 1;
        double paramTs = 0.2;
        double[][] rutaPrueba = generaRuta(200,0.25);
        carroOri.setPostura(rutaPrueba[2][0],rutaPrueba[2][1],rutaPrueba[2][2]+0.1,0);
//        for(int i=0;i<20;i++){
//            System.out.println(rutaPrueba[i][0] + " / " + rutaPrueba[i][1] + 
//                    "/" + rutaPrueba[i][2]);
//        }
                
        ControlPredictivo controlador = new ControlPredictivo(carroOri,rutaPrueba,
                                            horPredic,horCont,paramLanda,paramTs);
        controlador.calculaComando();
        System.exit(0);
        for (int i = 0; i < rutaPrueba.length; i++) {            
            double comandoVolante = controlador.calculaComando(); 
            if (comandoVolante > Math.PI/4)
                comandoVolante = Math.PI/4;
            if (comandoVolante < -Math.PI/4)
                comandoVolante = -Math.PI/4;
            System.out.println("Comando " + comandoVolante);
            carroOri.setConsignaVolante(comandoVolante);
            carroOri.calculaEvolucion(comandoVolante,2,0.2);
            int indice = UtilCalculos.indiceMasCercano(rutaPrueba,carroOri.getX(),carroOri.getY());
            double error = rutaPrueba[indice][2] - carroOri.getTita();
            System.out.println("Error " + error);
        }
//        double[] prediccion = controlador.calculaPrediccion(consVolante, vel);
//        for(int i=0;i<controlador.horPrediccion;i++){
//            System.out.println(prediccion[i]);
//        }
//        System.out.println("Angulo normalizado " +normalizaAngulo(0));
//        System.exit(0);
    }

		
}
