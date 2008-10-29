/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package sibtra.predictivo;

import Jama.Matrix;

/**
 * Esta clase implementa el algoritmo de control predictivo DMC
 * @author Jesus
 */
public class ControlPredictivo {
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
     * Dependiendo de su valor se le da más importancia a minimizar el error o 
     * a minimizar el gasto de comando. Si es igual a 0 no tiene en cuenta el gasto de 
     * comando, por lo que el controlador probablemente genere comandos bruscos.
     * En cambio si le damos un valor muy alto, restringimos mucho el uso de comando, 
     * con lo que se produce una actuación muy suave y lenta del controlador. Hay que 
     * alcanzar un término medio que permita actuar suficientemente rápido pero sin 
     * sobrepasamientos
     */
    double landa;
    double pesoError;
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
        carroSim = new Coche(carroOri);
        this.horPrediccion = horPrediccion;
        this.horControl = horControl;
        this.landa = landa;
        this.ruta = ruta;
        this.Ts = Ts;
        G = new Matrix(horPrediccion,horControl);
        this.landaEye = Matrix.identity(horControl,horControl).times(landa);
        prediccionPosicion = new double[horPrediccion][2];
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
        predicOrientacion = new double[horPrediccion];
        double[] vectorError = new double[horPrediccion];
        double[] orientacionesDeseadas = new double[horPrediccion];
        carroSim.copy(carroOriginal);//
        /*Hacemos la copia del objeto carroOriginal para realizar la simulación
         sobre la copia, de manera que no se modifiquen los datos reales*/
        //La siguiente linea de código es el cálculo del vector deseado en MAtlab
//        vec_deseado(1,:) = k_dist*(pos_ref(mod(ind_min(k)+cerca,length(pos_ref))+1,:) - [carro.x,carro.y])
//+ k_ang*[cos(tita_ref(mod(ind_min(k)+cerca,length(tita_ref))+1)),sin(tita_ref(mod(ind_min(k)+cerca,length(tita_ref))+1))];
        int indMin = calculaDistMin(ruta,carroSim.getX(),carroSim.getY());        
        double vectorDeseadoX = ruta[indMin][0] - carroSim.getX() + Math.cos(ruta[indMin][2]);
        double vectorDeseadoY = ruta[indMin][1] - carroSim.getY() + Math.sin(ruta[indMin][2]);
        orientacionesDeseadas[0] = Math.atan2(vectorDeseadoX,vectorDeseadoY);
        predicOrientacion[0] = carroSim.getTita();
        vectorError[0] = orientacionesDeseadas[0] - predicOrientacion[0];
        prediccionPosicion[0][0] = carroSim.getX();
        prediccionPosicion[0][1] = carroSim.getY();
        
        for (int i=1; i<horPrediccion;i++ ){
            carroSim.calculaEvolucion(comando,velocidad,Ts);
            predicOrientacion[i] = carroSim.getTita();
            indMin = calculaDistMin(ruta,carroSim.getX(),carroSim.getY());
            vectorDeseadoX = ruta[indMin][0] - carroSim.getX() + Math.cos(ruta[indMin][2]);
            vectorDeseadoY = ruta[indMin][1] - carroSim.getY() + Math.sin(ruta[indMin][2]);
            orientacionesDeseadas[i] = Math.atan2(vectorDeseadoY,vectorDeseadoX);
//            System.out.println("Indice minimo " + indMin);
//            System.out.println("Vector x "+vectorDeseadoX+" "+ "Vector y "+vectorDeseadoY+"\n");
//            System.out.println("Orientacion deseada " + orientacionesDeseadas[i] + " " 
//                                +"prediccion de orientacion " + predicOrientacion[i]+"\n");
            vectorError[i] = normalizaAngulo(orientacionesDeseadas[i] - predicOrientacion[i]);
            prediccionPosicion[i][0] = carroSim.getX();
            prediccionPosicion[i][1] = carroSim.getY();
        }
        
        return vectorError;
    }
    /**
     * Calcula la distancia mínima entre un punto y una trayectoria
     * @param ruta Array de doubles bidimensional
     * @param posX Coordenada x del punto
     * @param posY Coordenada y del punto
     * @return Índice de la ruta en el que se encuentra el punto de la 
     * ruta más cercano al punto (posX,posY)
     */
    public static int calculaDistMin(double[][] ruta,double posX,double posY){
        //Buscar punto más cercano al coche
            double distMin=Double.POSITIVE_INFINITY;
            int indMin=0;
            double dx;
            double dy;
            for(int i=0; i<ruta.length; i++) {
                dx=posX-ruta[i][0];
                dy=posY-ruta[i][1];
                double dist=Math.sqrt(dx*dx+dy*dy);
                if(dist<distMin) {
                    indMin=i;
                    distMin=dist;
                }
                
            }
            return indMin;
    }
    /**
     * Método optimizado de búsqueda del punto más cercano utilizando 
     * la información del último punto más cercano
     * @param indMinAnt
     * @return
     */
    private int calculaDistMinOptimizado(int indMinAnt){
        //TODO HAY QUE HACERLO!!
        return 0;
    }
    
    /**
     * Se le pasa un ángulo en radianes y devuelve ese mismo ángulo entre 
     * -PI y PI
     * @param angulo Ángulo a corregir
     * @return Ángulo en radianes corregido
     */
    static double normalizaAngulo(double angulo){
        angulo -= 2*Math.PI*Math.floor(angulo/(2*Math.PI));
        if (angulo >= Math.PI)
            angulo -= 2*Math.PI;
        return angulo;
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
        double[] respuestaEscalon = new double[horPrediccion];
        carroSim.setPostura(0,0,0,Math.PI/4);
        for (int i=0;i<horPrediccion;i++){
            carroSim.calculaEvolucion(Math.PI/4,velocidad, Ts);
            respuestaEscalon[i] = carroSim.getTita();
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
            int indice = ControlPredictivo.calculaDistMin(rutaPrueba,carroOri.getX(),carroOri.getY());
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
