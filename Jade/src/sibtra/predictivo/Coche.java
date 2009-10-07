/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package sibtra.predictivo;

import sibtra.util.Parametros;
import sibtra.util.UtilCalculos;
import Jama.Matrix;

/**
 * Modelo matemático del coche.
 * @author Jesus
 */
public class Coche implements Cloneable {
    // Matrices del espacio de los estados del volante
    
    protected Matrix A; 
    protected Matrix B;   
    protected Matrix C;
    protected Matrix D;
    /** Distancia entre eje ruedas posteriores y eje de rueda(s) delanteras.*/
    protected double longitud;
    /** Estado del modelo del volante. En esta representación corresponde a ángulo y velocidad angular del volante */
    protected Matrix estado;
    /** coordenad X de la posición del vehículo. Eje X en dirección Norte */
    protected double x = 0;
    /** Corrdenada Y de la posición del vehículo. Eje Y en dirección Oeste.*/
    protected double y = 0;
    /** Orientación del veículo con respecto al norte */
    protected double yaw = 0;
    /** Entrada al modelo de evolución del volante */
    protected double consignaVolante;
    protected double consignaVelocidad;
    /** Velocidad lineal del vehículo */
    protected double velocidad;
    
    /**
     * Copia los campos del objeto coche original sobre el objeto actual. 
     * No se duplican las matrices el espacio de los estados A,B,C,D, debido 
     * a que el objeto original y el actual deben de usar las mismas matrices.
     * El resto de campos se duplican (x,y,yaw y el estado)
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
        yaw = original.yaw;
        consignaVolante = original.consignaVolante;
        consignaVelocidad = original.consignaVelocidad;
        velocidad = original.velocidad;
        return this;
    }
    
    public Object clone(){
        Coche obj=null;
        try{
            obj=(Coche)super.clone();
        }catch(CloneNotSupportedException ex){
            System.out.println(" no se puede duplicar");
        }
        obj.estado=(Matrix)obj.estado.clone();
        return obj;
    }

    
    /**
     * Constructor por defecto. Asigna a las matrices del espacio 
     * de los estados el valor definido por defecto. También inicializa 
     * la longitud del vehículo a {@link Parametros.batalla} 
     */
    public Coche(){
    
//        double[][] arrayA = {{0.768181818181818,1.068181818181818}
//                            ,{-4.268181818181818,-3.568181818181818}};
//        double[] arrayB = {0.3,0.7};
//        double[] arrayC = {1.0,0.0};
//        double[] arrayD = {0.0};      
        double[][] arrayA = {{-1.578046155995488,0.118413876943000},
                            {0.078046155995488,-0.618413876943000}};
        double[] arrayB = {1.696460032938488,-0.696460032938488};
        double[][] arrayC = {{1.0 ,0.0}};
        double[] arrayD = {0};
        A = new Matrix(arrayA,2,2);
        B = new Matrix(arrayB,2);
//        C = new Matrix(arrayC,1,2);
        C = new Matrix(arrayC);
        C.print(10,	3);
        D = new Matrix(arrayD,1);
        estado = new Matrix(2,1);
        longitud =Parametros.batalla;
       
    }

    /**
      * Constructor que permite variar las matrices del espacio de los 
      * estados para variar el modelo de la dirección. También permite
      * variar la longitud del vehículo
     * @param matrixA Doble array de doubles 2x2
     * @param matrixB array de doubles 2x1
     * @param matrixC array de doubles 2x1
     * @param matrixD array de doubles 1x1
     * @param longi Longitud en metros del coche
     */
    public Coche(double[][] matrixA,double[] matrixB,double[][] matrixC,double[] matrixD,double longi){
    	//llamamos constructor por defecto
    	this();
    	if(matrixA==null || matrixA.length!=2 || matrixA[0].length!=2 
    			|| matrixB==null || matrixB.length!=2 
    			|| matrixC==null || matrixC.length!=2
    			|| matrixD==null || matrixD.length!=1)
    		throw new IllegalArgumentException("Matrices del sistema no tienen dimensiones correctas");
    	//modificamos con las matrices pasadas
        A = new Matrix(matrixA,2,2);
        B = new Matrix(matrixB,2);
        C = new Matrix(matrixC,1,2);
        D = new Matrix(matrixD,1);
        longitud = longi;
    }
    
    /** @return valor real del volante */
    public double getVolante(){       
        return C.times(estado).get(0,0);
    }
    /** @return valor real de la velocidad */
    public double getVelocidad(){
        return velocidad;
    }
    /** @return valor desado para el volante */
    public double getConsignaVolante(){
        return consignaVolante;
    }
    /** @return valor deseado para la velocidad */
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
    
    public double getYaw(){
        return yaw;
    }
    /**
     * Recoge la posición y orientación del vehículo. La fuente de la 
     * información puede ser la IMU y el GPS, el sistema odométrico
     * o incluso un programa de simulación
     * @param posX Coordenada local X
     * @param posY Coordenada loacal Y
     * @param orientacion Orientación del vehículo (no del volante) en radianes. (Entre -Pi y Pi) 
     */
    public void setPostura(double posX,double posY,double orientacion){
        x = posX;
        y = posY;
        yaw = orientacion;        
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
        yaw = orientacion;
        setVolante(posVolante);
    }
    /**
     * Recoge la posición y orientación del vehículo. La fuente de la 
     * información puede ser la IMU y el GPS, el sistema odométrico
     * o incluso un programa de simulación
     * @param posX Coordenada local X
     * @param posY Coordenada loacal Y
     * @param orientacion Orientación del vehículo (no del volante) en radianes. (Entre -Pi y Pi)s
     * @param posVolante posición del volante
     * @param velVolante velocidad del volante
     */
    public void setPostura(double posX,double posY,double orientacion,double posVolante, double velVolante){
        x = posX;
        y = posY;
        yaw = orientacion;
        setVolante(posVolante);
        estado.set(1,0,velVolante);
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
            
            
    public void setEstadoA0() {
    	estado.set(0, 0, 0.0);
    	estado.set(1, 0, 0.0);
    }

    
    /** Evoluciona el modelo de la bicicleta a partir de la posición del volante sacado de {@link #estado}
     * y de la {@link #velocidad}. Actuliazando los nuevos valores para {@link #x}, {@link #y} y {@link #yaw}.
     * @param Ts segundo de evolución a calcular
     */
    public void evolucionaBicicleta(double Ts) {
        //evoluciona el modelo de la bicicleta
        Matrix alfa = C.times(estado);
        double volante = alfa.get(0,0);
        //System.out.println("alfa escalar " + alfaEscalar);
        x = Math.cos(yaw)*velocidad*Ts + x;
        y = Math.sin(yaw)*velocidad*Ts + y;
        yaw = ((Math.tan(volante)/longitud)*velocidad*Ts + yaw);//%2*Math.PI
        yaw = UtilCalculos.normalizaAngulo(yaw);    	
    }

    /**
     * Calcula la evolución {@link #estado} del volante dada la {@link #consignaVolante}
     * @param Ts segundos de la evolución.
     */
    public void evolucionaVolante(double Ts) {
        //Evoluciona el volante
        Matrix estadoAux = A.times(estado).plus(B.times(this.consignaVolante)).times(Ts);
        estado.plusEquals(estadoAux);
    }
    
    /**
     * Calcula la evolución del vehículo en un instante de muestreo
     * @param volante Orientación del volante
     * @param velocidad Velocidad lineal del vehículo
     * @param Ts Periodo de muestreo
     */
    public void calculaEvolucion(double consignaVolante,double velocidad,double Ts){
        this.consignaVolante = consignaVolante;
        setConsignaVolante(consignaVolante);
        setVelocidad(velocidad);
        setConsignaVelocidad(velocidad); /*Por ahora no existe modelo dinámico para 
        calcular la evolución de la velocidad del vehículo, por lo que se supone
        la velocidad igual a la consigna de la velocidad*/

        calculaEvolucion(Ts);
        
    }
    
    /** Calcula la evolución del vehículo con los valores establecidos en los distintos campos
     * @param Ts segundos de la evolución
     */
    public void calculaEvolucion(double Ts) {
        evolucionaBicicleta(Ts);        
        evolucionaVolante(Ts);
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
//        double[][] arrayA = {{-2.8,-1.8182},{1.0,0}};
        System.out.println(UtilCalculos.normalizaAngulo(7.5));
        System.out.println(UtilCalculos.normalizaAngulo(-7.5));
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

	/**
	 * @param yaw the yaw to set
	 */
	public void setYaw(double yaw) {
		this.yaw = yaw;
	}

	/**
	 * @param volante the volante to set
	 */
	public void setVolante(double volante) {
		this.estado.set(0,0,volante);
	}

	/**
	 * @param x the x to set
	 */
	public void setX(double x) {
		this.x = x;
	}

	/**
	 * @param y the y to set
	 */
	public void setY(double y) {
		this.y = y;
	}
    
}
