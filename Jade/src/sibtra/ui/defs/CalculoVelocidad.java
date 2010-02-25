package sibtra.ui.defs;

/**
 * Lo debe implentar aquel {@link SubModulo} que sea capaz de generar consignas de velocidad.
 * 
 * @author alberto
 */ 

public interface CalculoVelocidad extends SubModulo {
	
	/** @return la consigna de velocida calculada */
	public double getConsignaVelocidad();
	

}
