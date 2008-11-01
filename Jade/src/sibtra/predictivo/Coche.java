/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package sibtra.predictivo;

import Jama.Matrix;

/**
 * Modelo matemático del coche.
 * @author Jesus
 */
public class Coche {
    // Matrices del espacio de los estados del volante
    
    protected Matrix A; 
    protected Matrix B;   
    protected Matrix C;
    protected Matrix D;
    protected double longitud;
    protected Matrix estado;
    protected double x = 0;
    protected double y = 0;
    protected double tita = 0;
    protected double consignaVolante;
    protected double consignaVelocidad;
    protected double volante;
    protected double velocidad;
    
    /**
     * Copia los campos del objeto coche original sobre el objeto actual. 
     * No se duplican las matrices el espacio de los estados A,B,C,D, debido 
     * a que el objeto original y el actual deben de usar las mismas matrices.
     * El resto de campos se duplican (x,y,tita y el estado)
     * @param original
     * @return Este objeto
     */
    public Coche copy(Coche original){
        A = original.A; 
        B = original.B;   
        C = original.C;
        D = original.D;
        longitud = original.longitud;
        if (original.estado != null)
            estado = (Matrix)original.estado.clone();
        x = original.x;
        y = original.y;
        tita = original.tita;
        consignaVolante = original.consignaVolante;
        consignaVelocidad = original.consignaVelocidad;
        volante = original.volante;
        velocidad = original.velocidad;
        return this;
    }
    /**
     * Constructor por defecto. Asigna a las matrices del espacio 
     * de los estados el valor definido por defecto. También inicializa 
     * la longitud del vehículo a 1.7 metros
     */
    public Coche(){
    
        double[][] arrayA = {{0.768181818181818,1.068181818181818}
                            ,{-4.268181818181818,-3.568181818181818}};
        double[] arrayB = {0.3,0.7};
        double[] arrayC = {1.0,0.0};
        double[] arrayD = {0.0};
        A = new Matrix(arrayA,2,2);
        B = new Matrix(arrayB,2);
        C = new Matrix(arrayC,2);
        D = new Matrix(arrayD,1);
        estado = new Matrix(2,1);
        longitud = 1.7;
       
    }
    public Coche(Coche original){
        this.copy(original);
    }
    /**
      * Constructor que permite variar las matrices del espacio de los 
      * estados para variar el modelo de la dirección. También permite
      * variar la longitud del vehículo
     * @param matrixA Doble array de doubles 2x2
     * @param matrixB array de doubles 2x1
     * @param matrixC array de doubles 2x1
     * @param matrixD array de doubles 2x1
     * @param longi Longitud en metros del coche
     */
    public Coche(double[][] matrixA,double[] matrixB,double[] matrixC,double[] matrixD,double longi){
        A = new Matrix(matrixA,2,2);
        B = new Matrix(matrixB,2);
        C = new Matrix(matrixC,2);
        D = new Matrix(matrixD,1);
        estado = new Matrix(2,1);
        longitud = longi;
    }
    public double getVolante(){       
        return volante;
    }
    public double getVelocidad(){
        return velocidad;
    }
    public double getConsignaVolante(){
        return consignaVolante;
    }
    public double getConsignaVelocidad(){
        return consignaVelocidad;
    }
    
    public Matrix getMatixA(){
        return A;
    }
    public Matrix getMatrixB(){
        return B;
    }
    public Matrix getMatrixC(){
        return C;
    }
    public Matrix getMatrixD(){
        return D;
    }
    public double getLongitud(){
        return longitud;
    }
    public Matrix getEstado(){
        return estado;
    }
    
    public double getX(){
        return x;
    }
    
    public double getY(){
        return y;
    }
    
    public double getTita(){
        return tita;
    }
    /**
     * Recoge la posición y orientación del vehículo. La fuente de la 
     * información puede ser la IMU y el GPS, el sistema odométrico
     * o incluso un programa de simulación
     * @param posX Coordenada local X
     * @param posY Coordenada loacal Y
     * @param orientacion Orientación del vehículo (no del volante) en radianes. (Entre -Pi y Pi)s
     * @param posVolante posición del volante
     */
    public void setPostura(double posX,double posY,double orientacion,double posVolante){
        x = posX;
        y = posY;
        tita = orientacion;
        volante = posVolante;
        estado.set(0,0,volante);
    }
    public void setVelocidad(double vel){
        velocidad = vel;    
    }
    public void setConsignaVelocidad(double consignaVelocidad){
        this.consignaVelocidad = consignaVelocidad;    
    }
    public void setConsignaVolante(double consignaVolante){
        this.consignaVolante = consignaVolante;
    }
            
            
    /**
     * Calcula la evolución del vehículo en un instante de muestreo
     * @param volante Orientación del volante
     * @param velocidad Velocidad lineal del vehículo
     * @param Ts Periodo de muestreo
     * @return Devuelve el ángulo del volante
     */
    public void calculaEvolucion(double consignaVolante,double velocidad,double Ts){
        this.consignaVolante = consignaVolante;
        this.velocidad = velocidad;
        consignaVelocidad = velocidad;/*Por ahora no existe modelo dinámico para 
        calcular la evolución de la velocidad del vehículo, por lo que se supone
        la velocidad igual a la consigna de la velocidad*/
        Matrix estadoAux = A.times(estado).plus(B.times(this.consignaVolante)).times(Ts);
        Matrix alfa = C.transpose().times(estado.plus(estadoAux));
        estado.plusEquals(estadoAux);
        volante = alfa.get(0,0);
        //System.out.println("alfa escalar " + alfaEscalar);
        x = Math.cos(tita)*velocidad*Ts + x;
        y = Math.sin(tita)*velocidad*Ts + y;
        tita = ((Math.tan(volante)/longitud)*velocidad*Ts + tita);//%2*Math.PI
        tita = normalizaAngulo(tita);
        

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
    /**
     * Calcula la evolución del vehículo en un instante de muestreo 
     * suponiendo velocidad cte = 1.5
     * @param volante Consigna para la orientación del volante
     * @param Ts Periodo de muestreo
     */
    public void calculaEvolucion(double volante,double Ts){
        
        calculaEvolucion(volante,1.5,Ts);
    }
    /**
     * Calcula la evolución del vehículo durante n instantes de muestreo
     * manteniendo constante el comando del volante
     * @param volante Consigna para la orientación del volante
     * @param velocidad Velocidad lineal del coche
     * @param Ts Periodo de muestreo
     * @param n Número de iteraciones
     * @return
     */
//    public Matrix calculaEvolucion(double volante,double velocidad,double Ts,int n){
//        
//        Matrix angulo = null;
//        for (int i = 0; i < n; i++) {
//            angulo = calculaEvolucion(volante,velocidad,Ts);
//            System.out.println("Ángulo");
//            angulo.print(1, 6);
//            System.out.println("Estado");
//            estado.print(1,6);
//            System.out.println("X " + x);
//            System.out.println("Y " + y);
//            System.out.println("TITA " + tita);
//        }
//        
//        return angulo;
//    }
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
//        double[][] arrayA = {{-2.8,-1.8182},{1.0,0}};
        System.out.println(normalizaAngulo(7.5));
        System.out.println(normalizaAngulo(-7.5));
        System.exit(0);
        Matrix A = new Matrix(2,2);
        Matrix B = new Matrix(2,1);
        Matrix C = new Matrix(1,2);
        double largo;
        double velocidad = 1;
        double comando = 0.4;
        double muestreo = 0.4;
        double[] vecX = new double[10];
        double[] vecY = new double[10];
        double[] vecTita = new double[10];
        int iteraciones = 5;
        Coche carro = new Coche();
//        A = carro.getMatixA();
//        B = carro.getMatrixB();
//        C = carro.getMatrixC();
//        A.print(1,1);
//        B.print(1,1);
//        C.print(1,1);
//        Matrix anguloVolante = carro.calculaEvolucion(comando,velocidad,muestreo,iteraciones);
//        Matrix state = carro.getEstado();
//        state.print(1,6);
//        for (int i =1; i<10; i++){
//            carro.calculaEvolucion(comando, velocidad, muestreo);
//            vecX[i] = carro.getX();
//            vecY[i] = carro.getY();
//            vecTita[i] = carro.getTita();
//        }
        

    }
    /*public recogeDatos(){
        
    }*/
    
}
