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
public class CocheModeloAntiguo extends Coche {
   
    /**
     * Constructor por defecto. Asigna a las matrices del espacio 
     * de los estados el valor definido por defecto. También inicializa 
     * la longitud del vehículo a 1.7 metros
     */
    public CocheModeloAntiguo(){
        super();
        double[][] arrayA = {{-2.8,-1.8182},{1.0,0}};
        double[] arrayB = {1.0,0};
        double[] arrayC = {0.30,1.8182};
        double[] arrayD = {0};
        A = new Matrix(arrayA,2,2);
        B = new Matrix(arrayB,2);
        C = new Matrix(arrayC,2);
        D = new Matrix(arrayD,1);
        estado = new Matrix(2,1);
        longitud = 1.7;
       
    }
    public CocheModeloAntiguo(Coche original){
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
    public CocheModeloAntiguo(double[][] matrixA,double[] matrixB,double[] matrixC,double[] matrixD,double longi){
        A = new Matrix(matrixA,2,2);
        B = new Matrix(matrixB,2);
        C = new Matrix(matrixC,2);
        D = new Matrix(matrixD,1);
        estado = new Matrix(2,1);
        longitud = longi;
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
    }
           
            
   
    
}
