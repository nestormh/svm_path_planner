package sibtra.odometria;

import sibtra.util.Parametros;

public class DatosOdometria {
	
	double xRel;	
	double yRel;
	double yaw;
	int contador;
	
	
	/**
	 * Constructor vacio. Se asigna NAN a los campos de la clase
	 */
	 
	
	public DatosOdometria(){
		this(Double.NaN,Double.NaN,Double.NaN);
		
	}
	/**
	 * Constructor don de se pasan los valores absolutos del vehículo
	 * @param x Coordenada x del vehículo
	 * @param y Coordenada y del vehículo
	 * @param yaw Orientación con respecto al norte geográfico del vehículo
	 */
	public DatosOdometria(double x,double y,double yaw){
		this.xRel = x;
		this.yRel = y;
		this.yaw = yaw;
	}
	/**
	 * Método que se encarga de obtener la posición relativa del vehículo así como su
	 * orientación relativa usando la información del sistema odométrico
	 * @param cuentasLeft Cuentas por periodo de muestreo del encoder de la rueda izquierda 
	 * @param cuentasRight Cuentas por periodo de muestreo del encoder de la rueda derecha
	 * @param Ts Periodo de muestreo del sistema odométrico en segundos
	 */
	public void calculaDatos (long cuentasLeft, long cuentasRight, double Ts){
		//Primero se calcula el factor de conversion entre el giro de la 
		//rueda y las cuentas del encoder
		double factorConversion = (Math.PI*Parametros.diametroRueda)/
		(Parametros.relacionReduccion*Parametros.resolucionEncoder);
		// Se calcula cuanto se han desplazado las ruedas
		double despLeft = factorConversion*cuentasLeft;
		double despRight = factorConversion*cuentasRight;
		// Se calcula el desplazamiento del centro del eje
		double despCentro = (despRight + despLeft)/2;
		// calculamos el incremento en la orientación del vehículo
		// recordar la aproximación del seno del ángulo por el ángulo 
		//cuando el incremento del ángulo es pequeño
		double incYaw = (despRight - despLeft)/Parametros.anchoCoche;
		this.yaw = (this.yaw + incYaw)%Math.PI;
		// Se aplica el modelo de la bicicleta (o algo así)
		this.xRel = this.xRel + despCentro*Math.cos(this.yaw)*Ts;
		this.yRel = this.yRel + despCentro*Math.sin(this.yaw)*Ts;
		contador++;
//		System.out.println("x "+xRel+ " y "+ yRel + " yaw " + yaw);
	}
	
	public void setYawXY(double yaw, double x, double y){
		this.xRel = x;
		this.yRel = y;
		this.yaw = yaw;
	}
	
	// Geters y Seters
	
	public int getContador() {
		return contador;
	}
	public void setContador(int contador) {
		this.contador = contador;
	}
	
	public double getxRel() {
		return xRel;
	}
	public void setxRel(double xRel) {
		this.xRel = xRel;
	}
	public double getyRel() {
		return yRel;
	}
	public void setyRel(double yRel) {
		this.yRel = yRel;
	}
	public double getYaw() {
		return yaw;
	}
	public void setYaw(double yaw) {
		this.yaw = yaw;
	}
}
