package sibtra.ui.defs;

/**
 * Lo debe implentar aquel {@link SubModulo} que sea capaz de detectar obstáculos.
 * 
 * @author alberto
 *
 */
public interface DetectaObstaculos extends SubModulo {
	
	/** @return distancia en metros libre de obstaculos */
	public double getDistanciaLibre();

}
