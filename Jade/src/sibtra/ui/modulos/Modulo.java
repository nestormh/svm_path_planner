/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.ui.VentanasMonitoriza;

/**
 * Interfaz base del resto de módulos
 * @author alberto
 *
 */
public interface Modulo {
	
	/** Para que las clases tengan constructor vacío, 
	 * con este método se realiza la inicializacion.
	 * @return si el módulo se pudo inicializar bien
	 */
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza);
	
	/** @return Descripcion breve del módulo */
	public String getDescripcion();
	
	/** @return nombre del módulo */
	public String getNombre();
	
	/** Cuando este método es invocado el módulo debe liberar todos los recursos.
	 * Hace las veces de un destructor.
	 */
	public void terminar();

}
