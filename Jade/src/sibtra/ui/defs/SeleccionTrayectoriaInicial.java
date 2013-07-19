package sibtra.ui.defs;

import sibtra.gps.Trayectoria;
import sibtra.ui.PanelTrayectoria;

/**
 * Interfaz que deben cumplir los modulos para la selección de una trayectoria inicial para la navegación.
 * <br>
 * Serán elegido por {@link PanelTrayectoria} para obtener una trayectoria inicial
 * 
 * @author alberto
 *
 */
public interface SeleccionTrayectoriaInicial extends Modulo {

	/** Se solicita una nueva ruta. Si el usuario decide no elegir ninguan se devuelve null*/
	public Trayectoria getTrayectoria();
	
}
