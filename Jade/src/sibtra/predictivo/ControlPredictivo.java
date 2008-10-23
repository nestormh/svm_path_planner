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
        double[] predicOrientacion = new double[horPrediccion];
        double[] vectorError = new double[horPrediccion];
        double[] orientacionesDeseadas = new double[horPrediccion];
        carroSim.copy(carroOriginal);//
        /*Hacemos la copia del objeto carroOriginal para realizar la simulación
         sobre la copia, de manera que no se modifiquen los datos reales*/
        int indMin = calculaDistMin(ruta,carroSim.getX(),carroSim.getY());        
        double vectorDeseadoX = ruta[indMin][0] - carroSim.getX();
        double vectorDeseadoY = ruta[indMin][1] - carroSim.getY();
        orientacionesDeseadas[0] = Math.atan2(vectorDeseadoX,vectorDeseadoY);
        predicOrientacion[0] = carroSim.getTita();
        vectorError[0] = orientacionesDeseadas[0] - predicOrientacion[0];
        
        for (int i=1; i<horPrediccion;i++ ){
            carroSim.calculaEvolucion(comando,velocidad,Ts);
            predicOrientacion[i] = carroSim.getTita();
            indMin = calculaDistMin(ruta,carroSim.getX(),carroSim.getY());
            vectorDeseadoX = ruta[indMin][0] - carroSim.getX();
            vectorDeseadoY = ruta[indMin][1] - carroSim.getY();
            orientacionesDeseadas[i] = Math.atan2(vectorDeseadoX,vectorDeseadoY);
            vectorError[i] = normalizaAngulo(orientacionesDeseadas[i] - predicOrientacion[i]);
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
    private static int calculaDistMin(double[][] ruta,double posX,double posY){
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
                                                   ,carroOriginal.getConsignaVelocidad());
        //    M = diag(peso_tita'*vector_error);
        Matrix M = new Matrix(vectorError,horPrediccion).transpose();
//        u = inv(G'*G + landa*eye(hor_control))*G'*M;
//    u_actual = u(1) + u_anterior;
//        %-----------Cálculo del vector de errores----------
        calculaG();
        Matrix Gt = G.transpose();
        Matrix vectorU = Gt.times(G).plus(landaEye).inverse().times(Gt).times(M);
        return vectorU.get(0,0) +  carroOriginal.getConsignaVolante();
       
    }
    
    private void calculaG(){
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
        
        
        
    }
    
    private double[] calculaEscalon(double velocidad){
        double[] respuestaEscalon = new double[horPrediccion];
        carroSim.setPostura(0,0,velocidad,Math.PI/4);
        for (int i=0;i<horPrediccion;i++){
            carroSim.calculaEvolucion(Math.PI/4,velocidad, Ts);
            respuestaEscalon[i] = carroSim.getTita();
        }
        return respuestaEscalon;
    }
    public void main(String[] args){
        
    }
    
}
