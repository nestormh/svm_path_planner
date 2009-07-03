/**
 * 
 */
package sibtra.ui.modulos;

import java.util.Vector;

import sibtra.gps.Ruta;

/**
 * @author alberto
 *
 */
public interface Motor extends Modulo {
	
	
	/** Cuando se invoca este método el método el motor debe empezar a actuar sobre el carro */
	public void actuar();
	
	/** Cunando este método es invocado el motor debe dejar de actuar sobre el carro */
	public void parar();

	/** Los calculadores, obstaculos, etc. solicitan la ruta a través de este método. 
	 * Si no hay ninguna seleccionada se tendrá que buscar a un selector de ruta para elegir una
	 * @return la ruta que se va a seguir  
	 */
	public Ruta getRutaSeleccionada();
	
	public void setCalculadorVelocidad(CalculoVelocidad calVel);
	
	public void setCalculadorDireccion(CalculoDireccion calDir);
	
	public void setDetectaObstaculos(DetectaObstaculos[] dectObs);
	
}
