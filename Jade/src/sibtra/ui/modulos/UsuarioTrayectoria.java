/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.gps.Trayectoria;
import sibtra.ui.VentanasMonitoriza;

/**
 * Deben cumplir aquellos que piden ruta a {@link VentanasMonitoriza} para que este les avise 
 * cuando la ruta se cambia.
 * @author alberto
 *
 */
public interface UsuarioTrayectoria {

	public void nuevaTrayectoria(Trayectoria tra);
	
}
