package sibtra.ui.modulos;

/**
 * Interfaz que deben cumplir los modulos para la selección de una ruta.
 * 
 * @author alberto
 *
 */
public interface SeleccionRuta extends Modulo {

	/** Se solicita una nueva ruta. Si el usuario decide no elegir ninguan se devuelve null*/
	public double[][] getTrayectoria();
	
}
