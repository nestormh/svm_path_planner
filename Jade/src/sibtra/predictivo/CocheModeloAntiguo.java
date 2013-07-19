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
        double[][] arrayC = {{0.30,1.8182}};
        double[] arrayD = {0};
        //Modelo para el motor nuevo sin realimentacion de la posicion del volante       
//        double[][] arrayA = {{-2.196460032938488,-0.966643893412244},{1.0,0}};
//        double[] arrayB = {1.0,0};
//        double[] arrayC = {1.696460032938488,0.966643893412244};
//        double[] arrayD = {0};
        A = new Matrix(arrayA,2,2);
        B = new Matrix(arrayB,2);
        C = new Matrix(arrayC);
        D = new Matrix(arrayD,1);
        estado = new Matrix(2,1);
        longitud = 1.7;
       
    }
    
	/**
	 * @param volante the volante to set
	 */
	public void setVolante(double volante) {
    	throw new IllegalArgumentException("En el modelo antiguo no es posible fijar la posición del volante");
    }
           
            
   
    
}
