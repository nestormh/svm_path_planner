package sibtra.lidar;


/**
 * Interfaz que debe cuplir cualquier <i>dispositivo</i> que 
 * sea capaz de obtener barrido laser 
 * 
 * @author alberto
 *
 */
public interface Lidar {

	/** @return el último barrido angular */
	public BarridoAngular getBarridoAngular();
	
	/**
	 * Suspende en {@link Thread} llamante hasta que se reciba un dato más actual que el pasado. 
	 * Si ya se tiene un dato más actual se vuelve inmediatamente.
	 * @param datoAnterior último dato que se tiene. Si se pasa <code>null</code> se espera hasta un nuevo dato
	 * @return el nuevo dato
	 */
	public BarridoAngular esperaNuevoBarrido(BarridoAngular datoAnterior);
	
}
