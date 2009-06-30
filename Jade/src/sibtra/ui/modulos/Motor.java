/**
 * 
 */
package sibtra.ui.modulos;

/**
 * @author alberto
 *
 */
public interface Motor extends Modulo {
	
	/** Cuando se invoca este método el método el motor debe empezar a actuar sobre el carro */
	public void actuar();
	
	/** Cunando este método es invocado el motor debe dejar de actuar sobre el carro */
	public void parar();

}
