/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.predictivo.Coche;


/**
 * @author alberto
 *
 */
public interface Motor extends Modulo {
	
	
	/** Cuando se invoca este método el método el motor debe empezar a actuar sobre el carro */
	public void actuar();
	
	/** Cunando este método es invocado el motor debe dejar de actuar sobre el carro */
	public void parar();

	
	public void setCalculadorVelocidad(CalculoVelocidad calVel);
	
	public void setCalculadorDireccion(CalculoDireccion calDir);
	
	public void setDetectaObstaculos(DetectaObstaculos[] dectObs);
	
	/** El motor debe llevar un modelo el coche que actualiza con cada nuevo dato de GPS, IMU, etc.
	 * Los modulos consultan el estado del coche a través de este modelo
	 */
	public Coche getModeloCoche();
	
}
