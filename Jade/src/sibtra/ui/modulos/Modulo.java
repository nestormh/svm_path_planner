/**
 * 
 */
package sibtra.ui.modulos;

/**
 * Interfaz base del resto de módulos
 * @author alberto
 *
 */
public interface Modulo {
	
	/** @return Descripcion breve del módulo */
	public String getDescripcion();
	
	/** @return nombre del módulo */
	public String getNombre();
	
	/** Cuando este método es invocado el módulo debe liberar todos los recursos.
	 * Hace las veces de un destructor.
	 */
	public void terminar();

}
